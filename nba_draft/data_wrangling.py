import requests
import pandas as pd
import time
import ftfy
from io import StringIO
import warnings
from bs4 import BeautifulSoup, Comment, MarkupResemblesLocatorWarning




def get_draft(start_yr=2015, end_yr=2024):
    """
    Scrapes first-round draft data for the NBA from Basketball Reference's website
    between the specified start and end years (inclusive).

    Args:
        start_yr (int): Starting year for draft data retrieval.
        end_yr (int): Ending year for draft data retrieval.

    Returns:
        pandas.DataFrame: A combined DataFrame containing draft information
                          for all years between start_yr and end_yr where a successful response was received.

    Raises:
      ValueError: If start_yr is not an integer or end_yr is not an integer.
      ValueError: If start_yr is after end_yr.
      ValueError: If start_yr < 1947 or end_yr > 2024.
      Exception: If no draft tables are found for any years in the specified range
                 or if a request fails with an error status code.

    Notes:
        - This function respects robots.txt and incorporates a one-second
          delay between requests.
        - It's recommended to use this function responsibly
          and adhere to the website's terms of service.
    """
    if not isinstance(start_yr, int) or not isinstance(end_yr, int):
      raise ValueError("start_yr and end_yr must be integers")
    if start_yr > end_yr:
      raise ValueError("start_yr cannot be after end_yr")
    if start_yr < 1947 or end_yr > 2024:
      raise ValueError("No data found before 1947 or after 2024")

    url1 = "https://www.basketball-reference.com/draft/"
    df_combined = None
    for i in range(start_yr, end_yr + 1):

      # Get html
      if i < 1950:
        html = requests.get(url1 + "BAA_" + str(i) + ".html")
      else:
        html = requests.get(url1 + "NBA_" + str(i) + ".html")
      if html.status_code != 200:
        raise Exception(f"Request failed with status code {html.status_code} for year {i}")
      soup = BeautifulSoup(html.text, features="html.parser")

      # Get table
      table = soup.find('div', class_ = 'table_wrapper').find('table', class_ = 'sortable stats_table')
      if table:
        df = pd.read_html(StringIO(str(table)))[0]
        df.columns = df.columns.droplevel(0)
        index_to_drop = df[df['Player'] == 'Round 2'].index
        df = df.drop(df.index[index_to_drop[0]:,])
        df['Draft_Yr'] = i

      # Combines tables
        if df_combined is None:
          df_combined = df
        else:
          df_combined = pd.concat([df_combined, df], ignore_index=True)

        time.sleep(1)

      if df_combined is None:
        raise Exception("No draft tables found for any years in the specified range.")

      # Fixes corrupted player names
      # df_combined['Players'] = df_combined['Players']
      df_combined = df_combined.dropna(subset=['Player'])
      df_combined["Player"] = df_combined["Player"].apply(ftfy.fix_text)

      
      # Changes duplicate column names
      finished_df = df_combined.set_axis(['Rk', 'Pk', 'Tm', 'Player', 'College', 'Yrs', 'G', 'MP_tot', 'PTS_tot', 'TRB_tot',
                  'AST_tot', 'FG%', '3P%', 'FT%', 'MP', 'PTS', 'TRB', 'AST', 'WS', 'WS/48',
                  'BPM', 'VORP', 'Draft_Yr'], axis=1)
      
    num_cols = ['Rk', 'Pk', 'Yrs', 'G', 'MP_tot', 'PTS_tot', 'TRB_tot',
                  'AST_tot', 'FG%', '3P%', 'FT%', 'MP', 'PTS', 'TRB', 'AST', 'WS', 'WS/48',
                  'BPM', 'VORP', 'Draft_Yr']
    for i in num_cols:
      finished_df[i] = pd.to_numeric(finished_df[i], errors='coerce')
    return finished_df



def add_colleges(dframe):

  """
    Adds college-related information (Win Percentage and Strength of Schedule) to the draft data frame.

    This function fetches college basketball data for each draft year from the Sports Reference website,
    processes the data to extract relevant metrics, and then merges this data with the draft data based on 
    the college names.

    The function performs the following steps:
    1. Collects and processes basketball team statistics for each draft year from the Sports Reference website.
    2. Creates a dictionary with processed data frames for each year containing team records and stats.
    3. Matches the college name from the input data frame with the corresponding team data for the draft year.
    4. Adds the college's Win Percentage (`WinPct_College`) and Strength of Schedule (`SOS_College`) to the input data frame.
    5. Handles missing or incorrectly formatted college names through a predefined mapping.

    Args:
        dframe (pandas.DataFrame): The input data frame containing draft data, including the 'College' and 'Draft_Yr' columns.

    Returns:
        pandas.DataFrame: A modified version of the input data frame with additional columns for 'WinPct_College' 
                          and 'SOS_College' representing the college's win percentage and strength of schedule, respectively.
    
    Raises:
        Exception: If the request to the Sports Reference website fails (non-200 status code).
    
    Notes:
        - The function assumes the presence of the 'College' and 'Draft_Yr' columns in the input data frame.
        - College names are mapped to their full names through a dictionary for better consistency. Colleges that do not appear on Sports-Reference have statistics recorded as NaN in dataframe.
        - The function suppresses warnings related to markup resembling locators in HTML content because tables are embedded in comments on source websites.
    """
  years = dframe['Draft_Yr'].unique()
  url = "https://www.sports-reference.com/cbb/seasons/men/"
  yrs_college = {}
  for year in years:
    full_url = f'{url}{year}-standings.html'

    # Get html code
    html = requests.get(full_url)
    if html.status_code != 200:
      raise Exception(f"Request failed with status code {html.status_code} for year {year}")

    soup = BeautifulSoup(html.text, 'html.parser')

    tables = []

    # (some of the tables are in commented sections in the html code)
    comments = soup.find_all(string=lambda text: isinstance(text, Comment))
    
    

    # Suppress only MarkupResemblesLocatorWarning
    warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

    for comment in comments:

      comment_soup = BeautifulSoup(comment, 'html.parser')
      tables_in_comment = comment_soup.find_all('table')
      tables.extend(tables_in_comment)


    cleaned_tables = []

    for html_table in tables: # getting all conference tables from the webpage
      # get the table into a pandas df
      html_string = str(html_table)
      df = pd.read_html(StringIO(html_string))[0]
      df.columns = df.columns.droplevel(0)
      df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
      cleaned_tables.append(df)

    stacked_df = pd.concat(cleaned_tables, ignore_index = True)
    finished_df = stacked_df.set_axis(['Rk', 'School', 'Conf', 'W', 'L', 'W-L%', 'Conf_W', 'Conf_L', 'Conf_W-L%', 'Own',
       'Opp', 'SRS', 'SOS', 'AP_Pre', 'AP_High', 'AP_Final', 'Notes'], axis=1)
    yrs_college[f"{year}"] = finished_df

  # add college information to draft table

  for i in range(len(dframe)):

    college_name = dframe['College'].iloc[i]
    college_dict = {
        'VCU': 'Virginia Commonwealth',
        'Central Florida': 'UCF',
        'IUPUI': 'IU Indy',
        'Central Michigan University': 'Central Michigan',
        'Rice University': 'Rice',
        'Western Carolina University': 'Western Carolina',
        'UConn': 'Connecticut',
        'Morehead State University': 'Morehead State',
        'Rider University': 'Rider',
        'SMU' : 'Southern Methodist',
        'UNC Charlotte' : 'Charlotte',
        'USC' : 'Southern California',
        'UW-Milwaukee' : 'Milwaukee',
        'BYU' : 'Brigham Young',
        'Cleveland State University' : 'Cleveland State',
        'Rutgers University' : 'Rutgers',
        'LSU' : 'Louisiana State',
        'Georgia State University' : 'Georgia State',
        'Loyola (MD)' : 'Loyola (MD)',
        'Miami (FL)': 'Miami (FL)',
        'UNLV' : 'Nevada-Las Vegas',
        'UNC' : 'North Carolina',
        "St. John's" : "St. John's (NY)",
        "Texas-El Paso" : "UTEP",
        "NYU" : "New York University",
        "Marist College" : "Marist",
        "Ole Miss" : "Mississippi",
        "Southern University and A&M College" : "Southern",
        "California State University, Los Angeles" : "Cal State Los Angeles",
        "Loyola Chicago" : "Loyola (IL)",
        "University of South Alabama" : "South Alabama",
        "United States Naval Academy" : "Navy",
        "Cal State Long Beach" : "Long Beach State",
        "UMass" : "Massachusetts",
        "University of Hartford" : "Hartford"
        }
    if not pd.isna(college_name):
      if college_name in college_dict:
        college_name = college_dict[f'{college_name}']
      elif " University" in college_name:
        college_name = college_name.replace(" University", "")

    if not pd.isna(college_name):
      college_name = college_name.strip()
      draft_year = dframe['Draft_Yr'].iloc[i]
      team_records = yrs_college[f"{draft_year}"]
      
    
      w_l_per = team_records.loc[team_records['School'] == college_name, 'W-L%']
      sos = team_records.loc[team_records['School'] == college_name, 'SOS']
      if not w_l_per.empty:
        dframe.loc[i, 'WinPct_College'] = w_l_per.iloc[0]
      else:
        dframe.loc[i, "WinPct_College"] = pd.NA

      if not sos.empty:
        dframe.loc[i, 'SOS_College'] = sos.iloc[0]
      else:
        dframe.loc[i, "SOS_College"] = pd.NA

  return dframe