import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(layout="wide")

# Function to forecast population using Prophet
def forecast_population_prophet(region_df, periods=5):
    model = Prophet(yearly_seasonality=False, daily_seasonality=False, weekly_seasonality=False)
    model.fit(region_df[['ds', 'y']])
    future = model.make_future_dataframe(periods=periods, freq='Y')
    forecast = model.predict(future)
    forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
    forecast_df['Type'] = ['Historical'] * len(region_df) + ['Forecast'] * periods
    return forecast_df

# Load the processed data
@st.cache_data
def load_data():
    # Replace 'infrastructure_data_processed.xlsx' with your actual file path
    df = pd.read_excel('everything.xlsx')  # Update this path accordingly
    df_long = df.melt(id_vars=['Область', 'Показатель'],
                      var_name='Год',
                      value_name='Значение')
    df_long.rename(columns={
        'Область': 'Region',
        'Показатель': 'Indicator',
        'Год': 'Year',
        'Значение': 'Value'
    }, inplace=True)
    df_pivot = df_long.pivot_table(index=['Region', 'Year'],
                                   columns='Indicator',
                                   values='Value').reset_index()
    return df_pivot

@st.cache_data
def load_population():
    # Replace 'infrastructure_data_processed.xlsx' with your actual file path
    df = pd.read_excel('everything.xlsx')  # Update this path accordingly
    df_long = df.melt(id_vars=['Область', 'Показатель'],
                      var_name='Год',
                      value_name='Значение')
    df_long.rename(columns={
        'Область': 'Region',
        'Показатель': 'Indicator',
        'Год': 'Year',
        'Значение': 'Value'
    }, inplace=True)
    df_pivot = df_long.pivot_table(index=['Region', 'Year'],
                                   columns='Indicator',
                                   values='Value').reset_index()
    # Prepare the population data
    population_indicator = 'Население'  # Replace with the exact column name if different
    df_population = df_pivot[['Region', 'Year', population_indicator]].copy()
    df_population.rename(columns={'Year': 'ds', population_indicator: 'y'}, inplace=True)
    df_population['ds'] = pd.to_datetime(df_population['ds'], format='%Y')
    df_population['Year'] = df_population['ds'].dt.year

    # Remove duplicates
    df_population = df_population.drop_duplicates(subset=['Region', 'Year'])

    return df_population

df_population = load_population()

# Forecasting for Все
@st.cache_data
def generate_forecasts(df_population, periods=5):
    forecast_results = []
    regions = df_population['Region'].unique()

    for region in regions:
        region_data = df_population[df_population['Region'] == region].copy()
        forecast_df = forecast_population_prophet(region_data, periods)
        forecast_df['Region'] = region
        forecast_df['Year'] = forecast_df['ds'].dt.year
        forecast_results.append(forecast_df)

    df_forecast = pd.concat(forecast_results, ignore_index=True)
    df_forecast.rename(columns={'yhat': 'Population Forecast'}, inplace=True)
    df_forecast.drop(columns=['ds'], inplace=True)
    return df_forecast

# Function to forecast investment gap using Prophet
def forecast_investment_gap(region_df, periods=5):
    model = Prophet(yearly_seasonality=False, daily_seasonality=False, weekly_seasonality=False)
    model.fit(region_df[['ds', 'y']])
    future = model.make_future_dataframe(periods=periods, freq='Y')
    forecast = model.predict(future)
    forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy() # Keep confidence intervals
    forecast_df['Type'] = ['Historical'] * len(region_df) + ['Forecast'] * periods
    return forecast_df

df_forecast = generate_forecasts(df_population)

# Merge historical and forecasted data
@st.cache_data
def merge_data(df_forecast, df_population):
    df_combined = df_forecast.merge(df_population, on=['Region', 'Year'], how='left')
    df_combined['Population'] = df_combined.apply(
        lambda row: row['y'] if row['Type'] == 'Historical' else row['Population Forecast'], axis=1
    )
    return df_combined

df_combined = merge_data(df_forecast, df_population)

# Streamlit App Layout
st.title("Демографические тренды регионов Казахстана")
st.write("Изменение населения играет важную роль в развитии регионов. Мы использовали простую модель прогнозирования для предсказания населения на будущие 5 лет (пакет prophet в Python). "
         "Существуют аномалии в виде областей которые были разделены в 2020-ых годах.")

# Sidebar for region selection and "Все" option
st.sidebar.header("Выбрать регион")
regions = df_combined['Region'].unique().tolist()
regions.insert(0, "Все")  # Add "Все" option at the beginning
selected_region = st.sidebar.selectbox("Region", regions)


if selected_region == "Все":
    fig = go.Figure()
    for region in df_combined['Region'].unique():
        region_data = df_combined[df_combined['Region'] == region].sort_values('Year')
        fig.add_trace(go.Scatter(
            x=region_data['Year'],
            y=region_data['Population'],
            mode='lines+markers',
            name=region,  # Display region name in legend
        ))
    fig.update_layout(
        title='Статистика населения по всем регионам',
        xaxis_title='Year',
        yaxis_title='Population',
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    # Filter data for the selected region
    region_data = df_combined[df_combined['Region'] == selected_region].sort_values('Year')

    # Create the plot
    fig = go.Figure()

    # Historical data
    historical = region_data[region_data['Type'] == 'Historical']
    fig.add_trace(go.Scatter(
        x=historical['Year'],
        y=historical['Population'],
        mode='lines+markers',
        name='Historical Population',
        line=dict(color='blue')
    ))

    # Forecasted data
    forecast = region_data[region_data['Type'] == 'Forecast']
    fig.add_trace(go.Scatter(
        x=forecast['Year'],
        y=forecast['Population Forecast'],
        mode='lines+markers',
        name='Forecasted Population',
        line=dict(color='red', dash='dash')
    ))

    # Confidence intervals (optional)
    fig.add_trace(go.Scatter(
        x=forecast['Year'],
        y=forecast['Population Forecast'] + (forecast['Population Forecast'] * 0.05),  # Example upper bound
        mode='lines',
        name='Upper Confidence Interval',
        line=dict(color='lightgrey'),
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=forecast['Year'],
        y=forecast['Population Forecast'] - (forecast['Population Forecast'] * 0.05),  # Example lower bound
        mode='lines',
        name='Lower Confidence Interval',
        line=dict(color='lightgrey'),
        fill='tonexty',
        fillcolor='rgba(0,100,80,0.2)',
        showlegend=False
    ))

    # Update layout
    fig.update_layout(
        title=f'Статистика населения для {selected_region}',
        xaxis_title='Year',
        yaxis_title='Population',
        hovermode='x unified'
    )

    # Display the plot
    st.plotly_chart(fig, use_container_width=True)

    # Display forecast details
    st.subheader("Forecast Details")
    if not forecast.empty:
        st.write(forecast[['Year', 'Population Forecast']])

    # Optionally, display raw data
    if st.checkbox("Show Raw Data"):
        st.subheader("Historical Population Data")
        st.write(historical[['Year', 'Population']])

        st.subheader("Forecasted Population Data")
        st.write(forecast[['Year', 'Population Forecast']])

# Investment Gap Line Chart with Forecast
st.subheader("Инвестиционный разрыв и Износ")
st.write("Мы решили ввести переменную, которую назвали Инвестиционный разрыв. Это показатель, который показывает разницу между инвестициями в основной капитал и суммой износа."
         "В свою очередь, сумма износа считается по формуле <<Наличие основных средств (начальная стоимость) * степень износа>>. "
         "По сути, это показатель чистого дефицита инвестиций для каждого региона - сумма, нужная для полного восстановления инфраструктуры. "
         "Мы также ипользовали линейную модель и прогнозировали инвестиционный разрыв на базе таких показателей как население, зарплата, "
         "сальдо миграции, износ, и различные ВРП показатели.")

col1, col2 = st.columns(2)
df_pivot = load_data()

# OLS Model Fitting
features = [
    'Население', 'Зарплата', 'Безработица', 'Сальдо миграции', 'Износ',
    'Доля ВВП', 'ВРП Промышленность', 'ВРП Энергия'
]
target = 'Инвестиционный разрыв'

ols_data = df_pivot[['Region', 'Year'] + features + [target]].copy()  # Include 'Region' here
ols_data.fillna(0, inplace=True)
for col in features + [target]:
    ols_data[col] = pd.to_numeric(ols_data[col], errors='coerce')
    ols_data[col].fillna(0, inplace=True)

X = ols_data[features]
y = ols_data[target]
X = sm.add_constant(X)
ols_model = sm.OLS(y, X).fit() # Fit the model once

if selected_region == "Все":
    fig_gap_all = go.Figure()
    for region in df_pivot['Region'].unique():
        investment_gap_data = df_pivot[df_pivot['Region'] == region][['Year', 'Инвестиционный разрыв']].copy()
        investment_gap_data.rename(columns={'Year': 'ds', 'Инвестиционный разрыв': 'y'}, inplace=True)
        investment_gap_data['ds'] = pd.to_datetime(investment_gap_data['ds'], format='%Y')
        investment_gap_data['y'].fillna(0, inplace=True)
        investment_gap_data['y'] = pd.to_numeric(investment_gap_data['y'], errors='coerce')
        investment_gap_data['y'].fillna(0, inplace=True)

        if len(investment_gap_data.dropna()) >= 2:
            forecast_gap = forecast_investment_gap(investment_gap_data)
            forecast_gap['Year'] = forecast_gap['ds'].dt.year

            fig_gap_all.add_trace(go.Scatter(
                x=forecast_gap['Year'],  # Use the full forecast data for x
                y=forecast_gap['yhat'],  # Use 'yhat' for both historical and forecast
                mode='lines+markers',
                name=region # Add region name to legend
            ))

    fig_gap_all.update_layout(
        title='Инвестиционный разрыв',  # Update title
        xaxis_title='Год',
        yaxis_title='Инвестиционный разрыв (mln KZT)',
        hovermode='x unified'
    )
    with col1:
        st.plotly_chart(fig_gap_all, use_container_width=True)

    fig_iznos_all = go.Figure()
    for region in df_pivot['Region'].unique(): # Iterate through regions for Iznos
        region_data_iznos = df_pivot[df_pivot['Region'] == region].copy()

        fig_iznos_all.add_trace(go.Scatter(
            x=region_data_iznos['Year'],
            y=region_data_iznos['Износ'],
            mode='lines+markers',
            name=f"{region} - Износ" # Distinguish regions in the legend
        ))

    fig_iznos_all.update_layout(
        title='Износ', # Updated title
        xaxis_title='Год',
        yaxis_title='Износ (%)',
        hovermode='x unified'
    )
    with col2:
        st.plotly_chart(fig_iznos_all, use_container_width=True)
else:
    region_ols_data = ols_data[ols_data['Region'] == selected_region].copy()

    # Forecast features (Example using rolling mean - replace with your preferred method)
    for feature in features:
        region_ols_data[feature + '_forecast'] = region_ols_data[feature].rolling(window=3, center=True).mean().shift(-1)

    future_X_region = region_ols_data[[feature + '_forecast' for feature in features]].copy()
    future_X_region = sm.add_constant(future_X_region.dropna())  # Handle NaNs

    forecast_gap_ols = ols_model.predict(future_X_region) # Use the pre-fitted OLS model

    fig_gap_ols = go.Figure()
    fig_gap_ols.add_trace(go.Scatter(
        x=region_ols_data['Year'],
        y=region_ols_data[target],
        mode='lines+markers',
        name='Исторический инвестиционный разрыв',
        line=dict(color='blue')
    ))

    fig_gap_ols.add_trace(go.Scatter(
        x=region_ols_data['Year'],
        y=forecast_gap_ols,
        mode='lines+markers',
        name='Предсказанный инвестиционный разрыв (OLS)',
        line=dict(color='green', dash='dash')
    ))

    fig_gap_ols.update_layout(
        title=f'Инвестиционный разрыв для {selected_region}',
        xaxis_title='Year',
        yaxis_title='Investment Gap (mln KZT)',
        hovermode='x unified',
    )
    with col1:
        st.plotly_chart(fig_gap_ols, use_container_width=False, width=600)

    fig_iznos = go.Figure()
    fig_iznos.add_trace(go.Scatter(
        x=region_ols_data['Year'],
        y=region_ols_data['Износ'],  # Plot historical "Износ"
        mode='lines+markers',
        name='Исторический износ',
        line=dict(color='purple')  # Example color
    ))

    # Forecast "Износ" - replace with your actual method
    iznos_forecast = region_ols_data['Износ'].rolling(window=3, center=True).mean().shift(-1)  # Example forecast

    fig_iznos.add_trace(go.Scatter(
        x=region_ols_data['Year'],  # You might need to adjust the x-axis for the forecast
        y=iznos_forecast,
        mode='lines+markers',
        name='Предсказанный износ',
        line=dict(color='orange', dash='dash') # Example color

    ))

    fig_iznos.update_layout(
        title=f'Износ для {selected_region}',
        xaxis_title='Год',
        yaxis_title='Износ (%)',  # Or appropriate units
        hovermode='x unified'
    )
    with col2:
        st.plotly_chart(fig_iznos, use_container_width=False, width=600)

# VDS by Industry Line Chart
st.subheader("ВДС по индустрии (нужно выбрать регион)")

if selected_region:  # Check if a region is selected
    region_industry_data = df_pivot[df_pivot['Region'] == selected_region]

    # Filter for VDS indicators (adjust this if your column names are different)
    vds_columns = [col for col in region_industry_data.columns if 'ВДС' in col and col != 'ВДС Промышленность']  # Exclude total VDS

    fig_vds = go.Figure()

    for industry_column in vds_columns:
        fig_vds.add_trace(go.Scatter(
            x=region_industry_data['Year'],
            y=region_industry_data[industry_column],
            mode='lines+markers',
            name=industry_column.replace('ВДС ', '')  # Remove "ВДС " for cleaner labels
        ))

    fig_vds.update_layout(
        title=f'ВДС по индустрии {selected_region}',
        xaxis_title='Год',
        yaxis_title='ВДС (mln tenge)',
        hovermode='x unified'  # Show data for all industries at the hovered year
    )

    st.plotly_chart(fig_vds, use_container_width=True)


### GDP Share Bar Chart ###
st.subheader("Доля ВВП по регионам")

selected_year = st.selectbox("Выбрать год", df_pivot['Year'].unique())  # Year selection

gdp_share_data = df_pivot[df_pivot['Year'] == selected_year].sort_values('Доля ВВП', ascending=False)

fig_gdp = go.Figure(data=[go.Bar(
    x=gdp_share_data['Region'],
    y=gdp_share_data['Доля ВВП'],
    marker_color='skyblue' # Customize bar color
)])

fig_gdp.update_layout(
    title=f'Доля ВВП в {selected_year}',
    xaxis_title='Region',
    yaxis_title='GDP Share (%)',
    xaxis={'categoryorder':'total descending'} # Sort bars in descending order
)

st.plotly_chart(fig_gdp, use_container_width=True)

# GDP Share Over Time Line Chart
st.subheader("Доли регионов в ВВП по годам")

# Create the line chart
fig_gdp_time = go.Figure()

for region in df_pivot['Region'].unique():
    region_gdp_data = df_pivot[df_pivot['Region'] == region].sort_values('Year')
    fig_gdp_time.add_trace(go.Scatter(
        x=region_gdp_data['Year'],
        y=region_gdp_data['Доля ВВП'],
        mode='lines+markers',
        name=region  # Show region name in the legend
    ))


fig_gdp_time.update_layout(
    title='Доля ВВП',
    xaxis_title='Год',
    yaxis_title='GDP Share (%)',
    hovermode='x unified' # Improved hover information
)

st.plotly_chart(fig_gdp_time, use_container_width=True)


st.write("Мы также решили посчитать индекс потребности в инфраструктуре. Этот показатель позволяет оценить, какие регионы нуждаются в большем внимании в плане инфраструктуры. "
         " Он рассчитывается на основе износа, основных средств и населения каждого региона с весами 0.7, 0.2, и 0.1 соответственно. ")

# Calculate Infrastructure Need Index (outside the interactive parts)
indicators = ['Износ', 'Основные средства (балансовая)', 'Население']
weights = {'Износ': 0.7, 'Основные средства (балансовая)': 0.2, 'Население': 0.1}  # Adjust weights as needed

data_for_index = df_pivot[indicators].copy()

# Standardize using MinMaxScaler
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data_for_index)
scaled_df = pd.DataFrame(scaled_data, columns=indicators)
scaled_df['Infrastructure_Need_Index'] = sum([scaled_df[indicator] * weights[indicator] for indicator in indicators])

# Merge the index back into df_pivot
df_pivot = pd.merge(df_pivot, scaled_df[['Infrastructure_Need_Index']], left_index=True, right_index=True, how='left')

if selected_region == "Все":
    fig_index_all = go.Figure()

    for region in df_pivot['Region'].unique():
        region_data = df_pivot[df_pivot['Region'] == region].copy()

        fig_index_all.add_trace(go.Scatter(
            x=region_data['Year'],
            y=region_data['Infrastructure_Need_Index'],
            mode='lines+markers',
            name=region
        ))

    fig_index_all.update_layout(
        title='Infrastructure Need Index',
        xaxis_title='Year',
        yaxis_title='Infrastructure Need Index',
        hovermode='x unified'
    )
    st.plotly_chart(fig_index_all, use_container_width=True)
elif selected_region:
    region_data = df_pivot[df_pivot['Region'] == selected_region].copy()

    # Calculate VRP Growth rate (year-over-year percentage change)
    region_data['VRP_Growth'] = region_data['ВРП Хозяйство'].pct_change() * 100

    fig_need_vs_vrp = go.Figure()

    fig_need_vs_vrp.add_trace(go.Scatter(
        x=region_data['Year'],
        y=region_data['Infrastructure_Need_Index'],
        mode='lines+markers',
        name='Infrastructure Need Index',
        yaxis='y1'  # Assign to the left y-axis
    ))

    fig_need_vs_vrp.add_trace(go.Scatter(
        x=region_data['Year'],
        y=region_data['VRP_Growth'],
        mode='lines+markers',
        name='VRP Growth (%)',
        yaxis='y2',  # Assign to the right y-axis
        line=dict(color='red')  # Different color for right y-axis
    ))

    fig_need_vs_vrp.update_layout(
        title=f'Infrastructure Need Index vs. ВРП рост для {selected_region}',
        xaxis_title='Year',
        yaxis_title='Infrastructure Need Index',
        yaxis2=dict(
            title='VRP Growth (%)',
            overlaying='y',
            side='right',
            color='red' # Same color for axis labels and line
        ),
        hovermode='x unified'
    )

    st.plotly_chart(fig_need_vs_vrp, use_container_width=True)