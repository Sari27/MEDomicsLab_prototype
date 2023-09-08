import dask.dataframe as dd
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from dask.diagnostics import ProgressBar
from transformers import AutoTokenizer, AutoModel, logging

ProgressBar().register()
logging.set_verbosity_error()
pd.set_option('mode.chained_assignment', None)

# Set biobert parameters
BIOBERT_PATH = '../extraction_models/pretrained_bert_tf/biobert_pretrain_output_all_notes_150000/'
BIOBERT_TOKENIZER = AutoTokenizer.from_pretrained(BIOBERT_PATH)
BIOBERT_MODEL = AutoModel.from_pretrained(BIOBERT_PATH)


def filter_dataframe_by_patient(df, id_column, patient_list):
    """
    Function used to filter the dataframe df by the patient list patient_list.

    :param df: Pandas dataframe.
    :param id_column: Column in df containing the patient identifiers.
    :param patient_list: The list of patient identifiers.

    :return: Filtered dataframe.

    """
    return df.loc[df[id_column].isin(patient_list)]


def convert_df_time_column_to_format(df, time_column, format='ISO8601'):
    """
    Convert the column time_column in the dataframe df to the format format.

    :param df: Pandas dataframe.
    :param time_column: Column to convert.
    :param format: Format in which we want to convert the time column.

    :return:

    """
    df[time_column] = dd.to_datetime(df[time_column], format=format)


def generate_ts_sr_embeddings(sr_patient_ts, ts_name):
    """
    Generate embeddings for an event given time series data.

    :param sr_patient_ts: Pandas time series concerning a certain event for a patient.
    :param ts_name: The event name (str).

    :return: sr_embeddings: Embeddings for the series sr_patient_event.

    """
    sr_embeddings = pd.Series(dtype='float64')
    if len(sr_patient_ts) > 0:
        sr_embeddings[ts_name + '_max'] = sr_patient_ts.max()
        sr_embeddings[ts_name + '_min'] = sr_patient_ts.min()
        sr_embeddings[ts_name + '_mean'] = sr_patient_ts.mean(skipna=True)
        sr_embeddings[ts_name + '_variance'] = sr_patient_ts.var(skipna=True)
        sr_embeddings[ts_name + '_meandiff'] = sr_patient_ts.diff().mean()
        sr_embeddings[ts_name + '_meanabsdiff'] = sr_patient_ts.diff().abs().mean()
        sr_embeddings[ts_name + '_maxdiff'] = sr_patient_ts.diff().abs().max()
        sr_embeddings[ts_name + '_sumabsdiff'] = sr_patient_ts.diff().abs().sum()
        sr_embeddings[ts_name + '_diff'] = sr_patient_ts.iloc[-1] - sr_patient_ts.iloc[0]
        # Compute the n_peaks
        peaks, _ = find_peaks(sr_patient_ts)
        sr_embeddings[ts_name + '_npeaks'] = len(peaks)
        # Compute the trend (linear slope)
        if len(sr_patient_ts) > 1:
            sr_embeddings[ts_name + '_trend'] = np.polyfit(np.arange(len(sr_patient_ts)), sr_patient_ts, 1)[0]
        else:
            sr_embeddings[ts_name + '_trend'] = 0
    return sr_embeddings


def generate_patient_ts_embeddings_between_dates(df_patient, time_column, ts_name_column, ts_value_column, ts_list,
                                                 ts_class_name, start_date, end_date):
    """
    Generate embeddings for time series list for a patient between 2 dates.

    :param df_patient: Pandas Dataframe of time series type for a patient.
    :param time_column: Column associated to time we want to consider in df_patient.
    :param ts_name_column: Column associated to time names values we want to consider in df_patient.
    :param ts_value_column: Column associated to time series values we want to consider in df_patient.
    :param ts_list: List of time series names we want to filter.
    :param ts_class_name: Name we want to give to the time series type in the master table.
    :param start_date: Date where we start looking for.
    :param end_date: Date where we stop looking for.

    :return: df_ts_embeddings: Generated embeddings for the time series in ts_list between start_date and end_date.

    """
    # Filter data by dates
    df_patient_between_dates = df_patient[
        (df_patient[time_column] >= start_date) & (df_patient[time_column] < end_date)]

    df_ts_embeddings = pd.DataFrame()
    for ts in ts_list:
        ts_name = ts_class_name + '_' + ts.lower().replace(',', '').replace(' -', '').replace(' ', '_')
        sr_ts = df_patient_between_dates[df_patient_between_dates[ts_name_column] == ts][ts_value_column].astype(float)
        sr_ts = generate_ts_sr_embeddings(sr_ts, ts_name)
        df_ts_embeddings = pd.concat([df_ts_embeddings, pd.DataFrame(sr_ts).transpose()], axis=1)
    if not df_ts_embeddings.empty:
        df_ts_embeddings.insert(0, 'Date', start_date)
    return df_ts_embeddings


def generate_patient_ts_embeddings(df_time_series, id_column_name, patient_id, ts_list, time_column, ts_name_column,
                                   ts_value_column, ts_class_name, timedelta, df_ts_key=None, key_column=None,
                                   name_column=None):
    """
    Generate embeddings for time series list for a patient.

    :param df_time_series: Pandas Dataframe of time series type.
    :param id_column_name: Column associated to patient identifiers in df_time_series.
    :param patient_id: Identifier of the patient we consider.
    :param ts_list: List of time series names we want to filter.
    :param time_column: Column associated to time we want to consider in df_patient.
    :param ts_name_column: Column associated to time names values we want to consider in df_patient.
    :param ts_value_column: Column associated to time series values we want to consider in df_patient.
    :param ts_class_name: Name we want to give to the time series type in the master table.
    :param timedelta: Range to consider getting timeseries.
    :param df_ts_key: Dataframe that links names from ts_list to their identifiers if necessary.
    :param key_column: Identifiers in df_ts_key.
    :param name_column: Names in df_ts_keys.

    :return: df_ts_embeddings: Generated embeddings for the time series in ts_list for the patient represented by
                               patient_id.

    """
    # Filter df_time_series with patient identifier
    df_patient = df_time_series.loc[df_time_series[id_column_name] == patient_id].sort_values(by=time_column)

    # Replace time series identifiers by time series name
    if not df_ts_key.empty:
        for key in df_patient[ts_name_column]:
            df_patient[ts_name_column].replace(key, df_ts_key.loc[df_ts_key[key_column] == key][name_column].iloc[0],
                                               inplace=True)

    # Create dataframe for embeddings
    df_ts_embeddings = pd.DataFrame()

    # Get embeddings every timedelta range
    start_date = df_patient[time_column].iloc[0]
    end_date = start_date + timedelta
    last_date = df_patient[time_column].iloc[-1]

    while start_date <= last_date:
        df_ts_embeddings = pd.concat([df_ts_embeddings,
                                      generate_patient_ts_embeddings_between_dates(df_patient, time_column,
                                                                                   ts_name_column, ts_value_column,
                                                                                   ts_list, ts_class_name, start_date,
                                                                                   end_date)], ignore_index=True)
        start_date += timedelta
        end_date += timedelta

    # Insert patient id
    df_ts_embeddings.insert(0, 'PatientID', patient_id)

    # Drop Nan values
    df_ts_embeddings.dropna(subset=df_ts_embeddings.columns[2:], how='all', inplace=True)
    return df_ts_embeddings


def generate_ts_embeddings(df_time_series, id_column_name, ts_list, time_column, ts_name_column, ts_value_column,
                           ts_class_name, timedelta, df_ts_key=None, key_column=None, name_column=None):
    """
    Generate time series embeddings.

    :param df_time_series: Pandas Dataframe of time series type.
    :param id_column_name: Column associated to patient identifiers in df_time_series.
    :param ts_list: List of time series names we want to filter.
    :param time_column: Column associated to time we want to consider in df_patient.
    :param ts_name_column: Column associated to time names values we want to consider in df_patient.
    :param ts_value_column: Column associated to time series values we want to consider in df_patient.
    :param ts_class_name: Name we want to give to the time series type in the master table.
    :param timedelta: Range to consider getting timeseries.
    :param df_ts_key: Dataframe that links names from ts_list to their identifiers if necessary.
    :param key_column: Identifiers in df_ts_key.
    :param name_column: Names in df_ts_keys.

    :return: df_ts_embeddings: Generated embeddings for the time series in ts_list.

    """
    df_ts_embeddings = pd.DataFrame()
    for patient_id in set(df_time_series[id_column_name]):
        df_ts_embeddings = pd.concat([df_ts_embeddings, generate_patient_ts_embeddings(df_time_series, id_column_name,
                                                                                       patient_id, ts_list, time_column,
                                                                                       ts_name_column, ts_value_column,
                                                                                       ts_class_name, timedelta,
                                                                                       df_ts_key, key_column,
                                                                                       name_column)], ignore_index=True)
    return df_ts_embeddings


def split_note_document(text, min_length=15):
    """
    Function taken from the GitHub repository of the HAIM study. Split a text if too long for embeddings generation.

    :param text: String of text to be processed into an embedding. BioBERT can only process a string with â‰¤ 512 tokens
           . If the input text exceeds this token count, we split it based on line breaks (driven from the discharge
           summary syntax).
    :param min_length: When parsing the text into its subsections, remove text strings below a minimum length. These are
           generally very short and encode minimal information (e.g. 'Name: ___').

    :return: chunk_parse: A list of "chunks", i.e. text strings, that breaks up the original text into strings with 512
             tokens.
             chunk_length: A list of the token counts for each "chunk".

    """
    tokens_list_0 = BIOBERT_TOKENIZER.tokenize(text)

    if len(tokens_list_0) <= 510:
        return [text], [1]

    chunk_parse = []
    chunk_length = []
    chunk = text

    # Go through text and aggregate in groups up to 510 tokens (+ padding)
    tokens_list = BIOBERT_TOKENIZER.tokenize(chunk)
    if len(tokens_list) >= 510:
        temp = chunk.split('\n')
        ind_start = 0
        len_sub = 0
        for i in range(len(temp)):
            temp_tk = BIOBERT_TOKENIZER.tokenize(temp[i])
            if len_sub + len(temp_tk) > 510:
                chunk_parse.append(' '.join(temp[ind_start:i]))
                chunk_length.append(len_sub)
                # reset for next chunk
                ind_start = i
                len_sub = len(temp_tk)
            else:
                len_sub += len(temp_tk)
    elif len(tokens_list) >= min_length:
        chunk_parse.append(chunk)
        chunk_length.append(len(tokens_list))

    return chunk_parse, chunk_length


def get_biobert_embeddings(text):
    """
    Function taken from the GitHub repository of the HAIM study. Obtain BioBERT embeddings of text string.

    :param text: Input text (str).

    :return: embeddings: Final Biobert embeddings with vector dimensionality = (1,768).
             hidden_embeddings: Last hidden layer in Biobert model with vector dimensionality = (token_size,768).

    """
    tokens_pt = BIOBERT_TOKENIZER(text, return_tensors="pt")
    outputs = BIOBERT_MODEL(**tokens_pt)
    last_hidden_state = outputs.last_hidden_state
    pooler_output = outputs.pooler_output
    hidden_embeddings = last_hidden_state.detach().numpy()
    embeddings = pooler_output.detach().numpy()

    return embeddings, hidden_embeddings


def get_biobert_embeddings_from_event_list(event_list, event_weights):
    """
    Function taken from the GitHub repository of the HAIM study. For notes obtain fixed-size BioBERT embeddings.

    :param event_list: Timebound ICU patient stay structure filtered by max_time_stamp or min_time_stamp if any.
    :param event_weights: Weights for aggregation of features in final embeddings.

    :return: aggregated_embeddings: BioBERT event features for all events.

    """
    event_weights_exp = []
    for idx, event_string in enumerate(event_list):
        weight = event_weights.values[idx]
        string_list, lengths = split_note_document(event_string)
        for idx_sub, event_string_sub in enumerate(string_list):
            # Extract biobert embedding
            embedding, hidden_embedding = get_biobert_embeddings(event_string_sub)
            # Concatenate
            if (idx == 0) & (idx_sub == 0):
                full_embedding = embedding
            else:
                full_embedding = np.concatenate((full_embedding, embedding), axis=0)
            event_weights_exp.append(weight)

    # Return the weighted average of embedding vector across temporal dimension
    try:
        aggregated_embedding = np.average(full_embedding, axis=0, weights=np.array(event_weights_exp))
    except:
        aggregated_embedding = np.zeros(768)

    return aggregated_embedding


def generate_patient_notes_embeddings_between_dates(df_patient, time_column, notes_column, first_visit_time, start_date,
                                                    end_date):
    """
    Generate patient notes embeddings between two dates.

    :param df_patient: Pandas Dataframe containing patient notes embeddings.
    :param time_column: Column name referring to the time column in df_patient.
    :param notes_column: Column name referring to the note column in df_patient.
    :param first_visit_time: The first patient visit date in all the recordings.
    :param start_date: Embeddings generation start at this date.
    :param end_date: Embeddings generation end at this date.

    :return: df_notes_embeddings: Pandas dataframe of the patient notes embeddings between start_date and end_date.

    """
    # Filter df_patient with start_date and end_date
    df_patient_between_dates = df_patient[(df_patient[time_column] >= start_date) &
                                          (df_patient[time_column] < end_date)]

    # Return if no value
    if df_patient_between_dates.empty:
        return

    # Get weights for embeddings
    df_patient_between_dates['deltacharttime'] = df_patient_between_dates[time_column].apply(
        lambda x: (x.replace(tzinfo=None) - first_visit_time).total_seconds() / 3600)

    # Call function for embeddings generation
    df_notes_embeddings = pd.DataFrame(
        [get_biobert_embeddings_from_event_list(df_patient_between_dates[notes_column],
                                                df_patient_between_dates['deltacharttime'])])
    df_notes_embeddings.insert(0, 'Date', start_date)

    return df_notes_embeddings


def generate_patient_notes_embeddings(df_notes, id_column_name, time_column, notes_column, first_visit_time, timedelta,
                                      patient_id):
    """
    Generate notes embeddings for a patient every timedelta range.

    :param df_notes: Pandas Dataframe containing notes for all the considered patients.
    :param id_column_name: Column referring to the patient identifiers in df_notes.
    :param time_column: Column name referring to the time column in df_patient.
    :param notes_column: Column name referring to the note column in df_patient.
    :param first_visit_time: The first patient (identified with patient_id) visit date in all the recordings.
    :param timedelta: Time range to consider for embeddings generation.
    :param patient_id: Identifier for the considered patient.

    :return: df_notes_embeddings: Pandas dataframe of the patient notes embeddings.

    """
    # Filter df_notes by patient id and sort by date
    df_patient = df_notes.loc[df_notes[id_column_name] == patient_id].sort_values(by=time_column)

    # Create dataframe for embeddings
    df_notes_embeddings = pd.DataFrame()

    # Get embeddings every timedelta range
    start_date = df_patient[time_column].iloc[0]
    end_date = start_date + timedelta
    last_date = df_patient[time_column].iloc[-1]

    # Get embeddings for every 24 hours range
    while start_date <= last_date:
        embeddings = generate_patient_notes_embeddings_between_dates(df_patient, time_column, notes_column,
                                                                     first_visit_time, start_date, end_date)
        df_notes_embeddings = pd.concat([df_notes_embeddings, embeddings], ignore_index=True)
        start_date += timedelta
        end_date += timedelta

    # Rename columns
    col_number = len(df_notes_embeddings.columns) - 1
    df_notes_embeddings.columns = ['Date'] + ['nrad_attr_' + str(i) for i in range(col_number)]

    # Insert patient haim_id in the dataframe
    df_notes_embeddings.insert(0, 'PatientID', patient_id)

    # Drop Nan values
    df_notes_embeddings.dropna(subset=df_notes_embeddings.columns[2:], how='all', inplace=True)

    return df_notes_embeddings


def generate_note_embeddings(df_notes, id_column, time_column, notes_column, sr_first_visit, timedelta):
    """
    Generate note embeddings for every patient in df_notes, every timedelta range.

    :param df_notes: Pandas Dataframe containing notes for all the considered patients.
    :param id_column: Column name referring to the patient identifiers in df_notes.
    :param time_column: Column name referring to the time column in df_patient.
    :param notes_column: Column name referring to the note column in df_patient.
    :param sr_first_visit: Series of the first visits for every patient. (sr[patient] = first_visit)
    :param timedelta: Time range to consider for embeddings generation.

    :return: df_notes_embeddings: Pandas dataframe of the all the patients notes embeddings.

    """
    df_notes_embeddings = pd.DataFrame()
    for patient_id in set(df_notes[id_column]):
        if patient_id in sr_first_visit.index:
            first_visit_time = sr_first_visit[patient_id]
            df_notes_embeddings = pd.concat([df_notes_embeddings, generate_patient_notes_embeddings(df_notes, id_column,
                                                                                                    time_column,
                                                                                                    notes_column,
                                                                                                    first_visit_time,
                                                                                                    timedelta,
                                                                                                    patient_id)],
                                            ignore_index=True)
    return df_notes_embeddings
