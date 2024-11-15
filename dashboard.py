import json

from prophet import Prophet
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
import json
import plotly.express as px
st.set_page_config(layout="wide")

@st.cache_data
def load_data():
    df = pd.read_excel('everything.xlsx')
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
    df = pd.read_excel('everything.xlsx')
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
    population_indicator = 'Население'
    df_population = df_pivot[['Region', 'Year', population_indicator]].copy()
    df_population.rename(columns={'Year': 'ds', population_indicator: 'y'}, inplace=True)
    df_population['ds'] = pd.to_datetime(df_population['ds'], format='%Y')
    df_population['Year'] = df_population['ds'].dt.year

    df_population = df_population.drop_duplicates(subset=['Region', 'Year'])

    return df_population

@st.cache_data
def generate_forecasts():
    # load from csv
    df_forecast = pd.read_csv('population_forecast.csv')
    return df_forecast

def forecast_investment_gap(region_df, periods=5):
    # load from csv
    # forecast_df = pd.read_csv('investment_gap_forecast.csv')
    # forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])

    # or

    model = Prophet(yearly_seasonality=False, daily_seasonality=False, weekly_seasonality=False)
    model.fit(region_df[['ds', 'y']])
    future = model.make_future_dataframe(periods=periods, freq='Y')
    forecast = model.predict(future)
    forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy() # Keep confidence intervals
    forecast_df['Type'] = ['Historical'] * len(region_df) + ['Forecast'] * periods
    return forecast_df

@st.cache_data
def merge_data(df_forecast, df_population):
    df_combined = df_forecast.merge(df_population, on=['Region', 'Year'], how='left')
    df_combined['Population'] = df_combined.apply(
        lambda row: row['y'] if row['Type'] == 'Historical' else row['Population Forecast'], axis=1
    )
    return df_combined

if __name__ == "__main__":
    df_population = load_population()
    df_forecast = generate_forecasts()
    df_combined = merge_data(df_forecast, df_population)
    left_margin, center_plot, right_margin = st.columns([1, 2, 1])

    st.title("Демографические тренды регионов Казахстана")
    st.write("Изменение населения играет важную роль в развитии регионов. Мы использовали простую модель прогнозирования для предсказания населения на будущие 5 лет (пакет prophet в Python). "
             "Существуют аномалии в виде областей которые были разделены в 2020-ых годах.")

    st.sidebar.header("Выбрать регион")
    regions = df_combined['Region'].unique().tolist()
    regions.insert(0, "Все")
    selected_region = st.sidebar.selectbox("Region", regions)

    if selected_region == "Все":
        fig = go.Figure()
        for region in df_combined['Region'].unique():
            region_data = df_combined[df_combined['Region'] == region].sort_values('Year')
            fig.add_trace(go.Scatter(
                x=region_data['Year'],
                y=region_data['Population'],
                mode='lines+markers',
                name=region,
            ))
        fig.update_layout(
            title='Статистика населения по всем регионам',
            xaxis_title='Year',
            yaxis_title='Population',
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        region_data = df_combined[df_combined['Region'] == selected_region].sort_values('Year')

        fig = go.Figure()

        historical = region_data[region_data['Type'] == 'Historical']
        fig.add_trace(go.Scatter(
            x=historical['Year'],
            y=historical['Population'],
            mode='lines+markers',
            name='Historical Population',
            line=dict(color='blue')
        ))

        forecast = region_data[region_data['Type'] == 'Forecast']
        fig.add_trace(go.Scatter(
            x=forecast['Year'],
            y=forecast['Population Forecast'],
            mode='lines+markers',
            name='Forecasted Population',
            line=dict(color='red', dash='dash')
        ))

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

        fig.update_layout(
            title=f'Статистика населения для {selected_region} области',
            xaxis_title='Year',
            yaxis_title='Population',
            hovermode='x unified'
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Forecast Details")
        if not forecast.empty:
            st.write(forecast[['Year', 'Population Forecast']])

    st.subheader("Инвестиционный разрыв и износ")
    st.write("Мы решили ввести переменную, которую назвали Инвестиционный разрыв. Это показатель, который показывает разницу между инвестициями в основной капитал и суммой износа."
             "В свою очередь, сумма износа считается по формуле <<Наличие основных средств (начальная стоимость) * степень износа>>. "
             "По сути, это показатель чистого дефицита инвестиций для каждого региона - сумма, нужная для полного восстановления инфраструктуры. "
             "Мы также ипользовали линейную модель и прогнозировали инвестиционный разрыв на базе таких показателей как население, зарплата, "
             "сальдо миграции, износ, и различные ВРП показатели.")

    col1, col2 = st.columns(2)
    df_pivot = load_data()

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
                    x=forecast_gap['Year'],
                    y=forecast_gap['yhat'],
                    mode='lines+markers',
                    name=region
                ))

        fig_gap_all.update_layout(
            title='Инвестиционный разрыв',
            xaxis_title='Год',
            yaxis_title='Инвестиционный разрыв (mln KZT)',
            hovermode='x unified'
        )
        with col1:
            st.plotly_chart(fig_gap_all, use_container_width=True)

        fig_iznos_all = go.Figure()
        for region in df_pivot['Region'].unique():
            region_data_iznos = df_pivot[df_pivot['Region'] == region].copy()

            fig_iznos_all.add_trace(go.Scatter(
                x=region_data_iznos['Year'],
                y=region_data_iznos['Износ'],
                mode='lines+markers',
                name=f"{region} - Износ"
            ))

        fig_iznos_all.update_layout(
            title='Износ',
            xaxis_title='Год',
            yaxis_title='Износ (%)',
            hovermode='x unified'
        )
        with col2:
            st.plotly_chart(fig_iznos_all, use_container_width=True)
    else:
        region_ols_data = ols_data[ols_data['Region'] == selected_region].copy()

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
            title=f'Инвестиционный разрыв для {selected_region} области',
            xaxis_title='Year',
            yaxis_title='Investment Gap (mln KZT)',
            hovermode='x unified',
        )
        with col1:
            st.plotly_chart(fig_gap_ols, use_container_width=False, width=600)

        fig_iznos = go.Figure()
        fig_iznos.add_trace(go.Scatter(
            x=region_ols_data['Year'],
            y=region_ols_data['Износ'],
            mode='lines+markers',
            name='Исторический износ',
            line=dict(color='purple')
        ))

        iznos_forecast = region_ols_data['Износ'].rolling(window=3, center=True).mean().shift(-1)

        fig_iznos.add_trace(go.Scatter(
            x=region_ols_data['Year'],
            y=iznos_forecast,
            mode='lines+markers',
            name='Предсказанный износ',
            line=dict(color='orange', dash='dash')

        ))

        fig_iznos.update_layout(
            title=f'Износ для {selected_region} области',
            xaxis_title='Год',
            yaxis_title='Износ (%)',
            hovermode='x unified'
        )
        with col2:
            st.plotly_chart(fig_iznos, use_container_width=False, width=600)

    st.subheader("ВДС по индустрии (нужно выбрать регион)")

    if selected_region:
        region_industry_data = df_pivot[df_pivot['Region'] == selected_region]

        vds_columns = [col for col in region_industry_data.columns if 'ВДС' in col and col != 'ВДС Промышленность']  # Exclude total VDS

        fig_vds = go.Figure()

        for industry_column in vds_columns:
            fig_vds.add_trace(go.Scatter(
                x=region_industry_data['Year'],
                y=region_industry_data[industry_column],
                mode='lines+markers',
                name=industry_column.replace('ВДС ', '')
            ))

        fig_vds.update_layout(
            title=f'ВДС по индустрии {selected_region}',
            xaxis_title='Год',
            yaxis_title='ВДС (mln tenge)',
            hovermode='x unified'
        )

        st.plotly_chart(fig_vds, use_container_width=True)


    ### GDP Share Bar Chart ###
    st.subheader("Доля ВВП по регионам")

    selected_year = st.selectbox("Выбрать год", df_pivot['Year'].unique())

    gdp_share_data = df_pivot[df_pivot['Year'] == selected_year].sort_values('Доля ВВП', ascending=False)

    fig_gdp = go.Figure(data=[go.Bar(
        x=gdp_share_data['Region'],
        y=gdp_share_data['Доля ВВП'],
        marker_color='skyblue'
    )])

    fig_gdp.update_layout(
        title=f'Доля ВВП в {selected_year}',
        xaxis_title='Region',
        yaxis_title='GDP Share (%)',
        xaxis={'categoryorder':'total descending'}
    )

    st.plotly_chart(fig_gdp, use_container_width=True)

    st.subheader("Доли регионов в ВВП по годам")

    fig_gdp_time = go.Figure()

    for region in df_pivot['Region'].unique():
        region_gdp_data = df_pivot[df_pivot['Region'] == region].sort_values('Year')
        fig_gdp_time.add_trace(go.Scatter(
            x=region_gdp_data['Year'],
            y=region_gdp_data['Доля ВВП'],
            mode='lines+markers',
            name=region
        ))


    fig_gdp_time.update_layout(
        title='Доля ВВП',
        xaxis_title='Год',
        yaxis_title='GDP Share (%)',
        hovermode='x unified'
    )

    st.plotly_chart(fig_gdp_time, use_container_width=True)

    st.subheader("Infrastructure Need Index")
    st.write("Мы также решили посчитать индекс потребности в инфраструктуре. Этот показатель позволяет оценить, какие регионы нуждаются в большем внимании в плане инфраструктуры. "
             " Он рассчитывается на основе нормализованных износа, основных средств и населения каждого региона с весами 0.85, 0.1, и 0.05 соответственно. ")

    indicators = ['Износ', 'Основные средства (балансовая)', 'Население']
    weights = {'Износ': 0.85, 'Основные средства (балансовая)': 0.1, 'Население': 0.05}

    data_for_index = df_pivot[indicators].copy()

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data_for_index)
    scaled_df = pd.DataFrame(scaled_data, columns=indicators)
    scaled_df['Infrastructure_Need_Index'] = sum([scaled_df[indicator] * weights[indicator] for indicator in indicators])

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

        region_data['VRP_Growth'] = region_data['ВРП Хозяйство'].pct_change() * 100

        fig_need_vs_vrp = go.Figure()

        fig_need_vs_vrp.add_trace(go.Scatter(
            x=region_data['Year'],
            y=region_data['Infrastructure_Need_Index'],
            mode='lines+markers',
            name='Infrastructure Need Index',
            yaxis='y1'
        ))

        fig_need_vs_vrp.add_trace(go.Scatter(
            x=region_data['Year'],
            y=region_data['VRP_Growth'],
            mode='lines+markers',
            name='VRP Growth (%)',
            yaxis='y2',
            line=dict(color='red')
        ))

        fig_need_vs_vrp.update_layout(
            title=f'Infrastructure Need Index vs. ВРП рост для {selected_region} области',
            xaxis_title='Year',
            yaxis_title='Infrastructure Need Index',
            yaxis2=dict(
                title='VRP Growth (%)',
                overlaying='y',
                side='right',
                color='red'
            ),
            hovermode='x unified'
        )

        st.plotly_chart(fig_need_vs_vrp, use_container_width=True)

    # Investment Priority Index
    st.subheader("Investment Priority Index")
    st.write("Мы решили далее экспериментировать с возможными метриками для оценки приоритетов инвестиций в региональную инфраструктуру. Так мы решили ввести показатель, который назвали Индекс приоритета инвестиций. "
             "Его отличие от индекса потребности в инфраструктуре в том, что он взвешен по доле ВВП региона. Таким образом, мы можем оценить, какие регионы нуждаются в большем внимании в плане инфраструктуры также являясь важной составляющей экономики.")

    # Calculate Investment Priority Index (outside interactive parts)
    df_pivot['GDP_Share_Scaled'] = MinMaxScaler().fit_transform(df_pivot[['Доля ВВП']]) 
    df_pivot['Investment_Priority_Index'] = df_pivot['Infrastructure_Need_Index'] * df_pivot['GDP_Share_Scaled']


    if selected_region == "Все":
        fig_priority_all = go.Figure()

        for region in df_pivot['Region'].unique():
            region_data = df_pivot[df_pivot['Region'] == region].copy()

            fig_priority_all.add_trace(go.Scatter(
                x=region_data['Year'],
                y=region_data['Investment_Priority_Index'],
                mode='lines+markers',
                name=region
            ))

        fig_priority_all.update_layout(
            title='Investment Priority Index for All Regions',
            xaxis_title='Year',
            yaxis_title='Investment Priority Index',
            hovermode='x unified'
        )

        st.plotly_chart(fig_priority_all, use_container_width=True)


    elif selected_region:
        region_data = df_pivot[df_pivot['Region'] == selected_region].copy()

        fig_priority = go.Figure()

        fig_priority.add_trace(go.Scatter(
            x=region_data['Year'],
            y=region_data['Investment_Priority_Index'],
            mode='lines+markers',
            name='Investment Priority Index'
        ))

        fig_priority.update_layout(
            title=f'Investment Priority Index for {selected_region}',
            xaxis_title='Year',
            yaxis_title='Investment Priority Index',
            hovermode='x unified'
        )

        # Change width of the figure
        st.plotly_chart(fig_priority, use_container_width=True)

    # Function to create the animated map
    def create_animated_map(df, geojson_data, featureidkey, indicator_col):
        fig = px.choropleth_mapbox(
            df,
            geojson=geojson_data,
            locations='Region',
            featureidkey=featureidkey,
            color=indicator_col,
            color_continuous_scale="Turbo",  # Customizable
            mapbox_style="carto-positron",
            zoom=3.2,
            center={"lat": 50, "lon": 68},  # Approx. center of Kazakhstan
            opacity=0.5,
            labels={indicator_col: indicator_col},
            animation_frame="Year",
            height=600,
            width=700
        )
        fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})  # Remove margins
        return fig

    st.subheader("Степень износа по регионам в разные года")
    st.write("Мы также решили визуализировать степень износа по регионам в разные года, а далее индекс потребности в инвестициях. "
             "Мы выделили для себя несколько областей, которые нуждаются в большем внимании в плане дополнительных инвестиций в инфраструктуру:"
             " Атырауская, Карагандинская, Актюбинская.")

    col3, col4 = st.columns(2)

    with open("kz.json", encoding="utf-8") as f:
        geojson_data = json.load(f)

    featureidkey = "properties.name"

    if selected_region == "Все":
        map_df = df_pivot.copy()
    else:
        map_df = df_pivot[df_pivot['Region'] == selected_region].copy()

    dataframe_region_names = df_pivot['Region'].unique().tolist()
    print("DataFrame Region Names:", dataframe_region_names)

    geojson_region_names = [feature['properties']['name'] for feature in geojson_data['features']]
    print("GeoJSON Region Names:", geojson_region_names)

    if selected_region == "Все":
        map_df_priority = df_pivot.copy()
    else:
        map_df_priority = df_pivot[df_pivot['Region'] == selected_region].copy()

    # Ensure that Investment_Priority_Index is numeric
    map_df_priority['Investment_Priority_Index'] = pd.to_numeric(map_df_priority['Investment_Priority_Index'], errors='coerce')

    # Normalize Investment_Priority_Index
    map_df_priority['Investment_Priority_Index'] = MinMaxScaler().fit_transform(map_df_priority[['Investment_Priority_Index']])

    # Drop rows with missing values in Investment_Priority_Index
    map_df_priority = map_df_priority.dropna(subset=['Investment_Priority_Index'])

    with col3:
        animated_fig = create_animated_map(map_df, geojson_data, featureidkey, "Износ")
        st.plotly_chart(animated_fig, use_container_width=True)

    with col4:
        animated_fig_priority = create_animated_map(map_df_priority, geojson_data, featureidkey, "Investment_Priority_Index")
        st.plotly_chart(animated_fig_priority, use_container_width=True)
