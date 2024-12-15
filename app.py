# Here is some starter code to get the data:
import streamlit as st
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from nba_draft import *



@st.cache_data
def get_data(start_year, end_year):
  data = get_draft(start_year, end_year)
  return data

@st.cache_data
def adding_colls(dataframe):
   added_colleges = add_colleges(dataframe)
   return added_colleges


st.title("NBA Draft and College Info")

if 'draft_df' not in st.session_state:
    st.session_state.draft_df = None

# Create a sidebar for the date range filter
with st.sidebar:
    st.header("Date Range Filter")

    # Use date input widgets for start and end years
    start_year = st.number_input("Start Year", min_value=1947, max_value=2024, value=2020)
    end_year = st.number_input("End Year", min_value=1947, max_value=2024, value=2024)

# Trigger data scraping and display on button click
if st.button("Scrape and Display"):
    if start_year < 1950 or end_year < 1950:
        st.text("***If accessing data before 1950, college win percentage and strength of schedule will not be included in dataframe***")
        st.session_state.draft_df = get_data(start_year, end_year)
    else:
        st.session_state.draft_df = adding_colls(get_data(start_year, end_year))

if st.session_state.draft_df is not None:
    # Display the dataframe
    st.dataframe(st.session_state.draft_df)

    # Response Variable Selection
    if start_year < 1950 or end_year < 1950:
      options = ['Yrs', 'G', 'MP_tot', 'PTS_tot', 'TRB_tot',
               'AST_tot', 'FG%', '3P%', 'FT%', 'MP', 'PTS', 'TRB', 'AST', 'WS', 'WS/48',
               'BPM', 'VORP']
    else:
      options = ['Yrs', 'G', 'MP_tot', 'PTS_tot', 'TRB_tot',
               'AST_tot', 'FG%', '3P%', 'FT%', 'MP', 'PTS', 'TRB', 'AST', 'WS', 'WS/48',
               'BPM', 'VORP', 'WinPct_College', 'SOS_College']

    selected_var = st.selectbox('Select a Response Variable', options)

    st.write('You selected:', selected_var)

    # Initialize figure
    fig = go.Figure()

    # Get unique years
    draft_years = st.session_state.draft_df['Draft_Yr'].unique()

    for year in draft_years:
        # Filter data for the year
        year_data = st.session_state.draft_df[st.session_state.draft_df['Draft_Yr'] == year]

       
        # Drop numeric NA rows
        #year_data['Pk'] = pd.to_numeric(year_data['Pk'], errors='coerce')
        #year_data[selected_var] = pd.to_numeric(year_data[selected_var], errors='coerce')
        year_data = year_data.dropna(subset=['Pk', selected_var])  # Drop rows with non-convertible values

        # Scatter points for this year
        fig.add_trace(go.Scatter(
            x=year_data['Pk'],
            y=year_data[selected_var],
            mode='markers',
            name=f"Year {year}",
            marker=dict(size=6)
        ))

        # Fit linear regression for this year
        if len(year_data) > 1:  # Ensure enough points for regression
            x = year_data['Pk'].values.reshape(-1, 1)
            y = year_data[selected_var].values
            reg = LinearRegression().fit(x, y)
            line_x = np.linspace(min(year_data['Pk']), max(year_data['Pk']), 100)
            line_y = reg.predict(line_x.reshape(-1, 1))

            # Add regression line for this year
            fig.add_trace(go.Scatter(
                x=line_x,
                y=line_y,
                mode='lines',
                name=f"Trend {year}",
                line=dict(width=2)
            ))

    # Update layout
    fig.update_layout(
        title=f'Draft Pick vs. {selected_var} with Regression Lines by Draft Year',
        xaxis_title='Pick Number',
        yaxis_title=selected_var,
        legend_title="Legend"
    )

    st.plotly_chart(fig)

    st.header("Draft Position Analysis")
    st.write("Predicts player draft position using your dataframe")

    knn = knn_analysis(st.session_state.draft_df)
    linreg = linreg_analysis(st.session_state.draft_df)
    dtree = dtree_analysis(st.session_state.draft_df)

    # Dropdown (or radio) for selecting the model
    model_choice = st.selectbox(
    "Select a model to view performance:",
    ["K Nearest Neighbor", "Linear Regression", "Decision Tree"]
)

# Display the corresponding model's output based on selection
    if model_choice == "K Nearest Neighbor":
        st.write(f"Model accuracy without college info: {round(knn[0], 4)}")
        st.write(f"Model accuracy with college info: {round(knn[1], 4)}")
    elif model_choice == "Linear Regression":
        st.write(f"Model R-squared without college info: {round(linreg[0], 4)}")
        st.write(f"Model R-squared with college info: {round(linreg[1], 4)}")
    elif model_choice == "Decision Tree":
        st.write(f"Model accuracy without college info: {round(dtree[0], 4)}")
        st.write(f"Model accuracy with college info: {round(dtree[1], 4)}")
