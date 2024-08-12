import streamlit as st
import pandas as pd
from reguler_fuzzy import ChengReguler
from ga_fuzzy import ChengGA


def forecast_reguler_fuzzy():

    # Membuat Class ChengReguler yang berisi function function untuk prediksi data
    # class ini mengembalikan kelas cheng yang telah berhasil memprediksi
    cheng = ChengReguler({
        "data": train_data,
        "ga": False,
        "gen": None
    })

    # function di streamlit untuk menampilkan line chart
    # Chart ini untuk membandingkan data aktual dan data prediksi dari fuzzy time series reguler
    st.line_chart(
        pd.DataFrame(
            {
                "actual": train_data,
                "forecast": cheng.forecast_result
            }),
        color=["#FF0000", "#0000FF"]
    )

    # Mengembalikan nilai mape, prediksi, dan error setiap data yang di prediksi
    return cheng.mape, cheng.forecast_result, cheng.forecast_error



def forecast_ga_fuzzy():

    # membuat placeholder untuk menampilkan real time MAPE graphic history di streamli
    mape_graph_placeholder = st.empty()

    # Membuat class ChengGA yang berisi function untuk melakukan training dan prediksi Fuzzy Cheng Dengan GA Optimasi
    # Data Parameter didapatkan  dari input parameter streankut
    cheng_ga = ChengGA({
        "data": train_data,
        "num_generations":iteration,
        "num_chromosome":chromosome,
        "mutation":mutation,
        "crossover":crossover,
        "graph": mape_graph_placeholder
    })

    # Mmemanggil fungsi run untuk melakukan training dan prediksi
    # Fungsi ini mengembalikan hasil prediksi tiap data dan error tiap data
    result, error = cheng_ga.run()

    # Save History dari MAPE
    pd.DataFrame(
        {
            "mape":cheng_ga.history
        }
    ).to_csv(f"History/{iteration}Iteration_{chromosome}Chromosome_{crossover}Crossover_{mutation}Mutation.csv")

    # Menampilkan grafik perbandingan antara data aktual dan data prediksi dari Cheng Fuzzy GA
    st.line_chart(
        pd.DataFrame(
            {
                "actual": train_data,
                "forecast": result
            }),
        color=["#FF0000", "#0000FF"]
    )

    # Mengembalikan nilai mape, prediksi setiap data, dan error prediksi setiap data
    return cheng_ga.history[len(cheng_ga.history) -1], result, error

st.set_page_config(page_title='Fuzzy Time Series Optimization', layout='wide')
st.title('Brent Oil Price Forecasting')

st.sidebar.title('Menu')

st.subheader('Data Upload')

# Fungsi untuk upload file
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

# Jika file sudah di upload, eksekusi kode di dalam block ini
if uploaded_file is not None:

    # Baca file csv yang diupload
    df = pd.read_csv(uploaded_file)

    # Ubah nama column menjadi data dan price
    df.columns = ['Date', 'Price']

    # Merubah tipe data pada column date menjadi tipe data tanggal
    df['Date'] = pd.to_datetime(df['Date'])

    # Block kode untuk mendapatkan parameter input dari user
    st.subheader('Parameter Input')
    start_year = st.slider('Select the start year:', min_value=2000, max_value=2023, value=2010, step=1)
    iteration = st.number_input('Enter Iteration:', min_value=1, value=1)
    chromosome = st.number_input('Enter Chromosome:', min_value=3, value=3)
    mutation = st.number_input('Enter Mutation:', min_value=0.0, max_value = 1.0, value=0.5, step=0.0001)
    crossover = st.number_input('Enter Crossover:', min_value=0.0, max_value = 1.0, value= 0.5)

    # Potong daa menjadi pilihan user
    df = df[df['Date'] >= f'{start_year}-01-01']
    st.subheader("Price History Chart")

    # Splitting data
    price_data = df['Price'].to_list()
    split_index = int(len(price_data) * 0.75)

    train_data = price_data[:split_index]
    test_data = price_data[split_index:]
    test_data.insert(0, train_data[len(train_data) - 1])

    # Menampilkan data yang sudah di potong
    st.line_chart(train_data)
    st.write(df)

    # Jika user menekan tombol submit maka eksekusi kode di bawah ini
    if st.button('Submit'):
        st.subheader("Fuzzy Cheng Prediction")

        # Training dan Prediksi CHeng Fuzzy ( Fungsi yang di atas tadi)
        rm, forecast_cheng, cheng_error = forecast_reguler_fuzzy()

        st.subheader("Fuzzy Cheng GA Prediction")

        # Training dan Prediksi Cheng Fuzzy Optimasi dengan Genetik ALgoritma ( FUngsi yang di atas tadi)
        gam, forecast_cheng_ga, cheng_ga_error = forecast_ga_fuzzy()

        # comparison_df = pd.DataFrame({
        #     'Reguler Fuzzy': [rm, 0],
        #     'GA Fuzzy':   [0, gam]
        # })

        st.subheader("MAPE Comparison")
        # st.bar_chart(comparison_df)

        # Dataframe Komparasi MAPE
        comparison_df = pd.DataFrame({
            'Tanggal': df['Date'].to_list()[:split_index],
            'Harga':   df['Price'].to_list()[:split_index],
            'Prediksi Cheng' : forecast_cheng,
            'Prediksi Cheng Algen' : forecast_cheng_ga,
            'Error Cheng' : cheng_error,
            'Error Cheng Algen' : cheng_ga_error
        })
        # df_show = train_data.assign(PrediksiCheng=forecast_cheng)
        st.dataframe(comparison_df, use_container_width=True)
        st.subheader("Hasil")

        # Menampilkan MAPE
        st.write(f"MAPE Cheng :{rm}")
        st.write(f"MAPE Cheng Algen :{gam}")

        # Export data prediksi dari kedua algoritma
        comparison_df.to_csv(f"Prediction/{iteration}Iteration_{chromosome}Chromosome_{crossover}Crossover_{mutation}Mutation.csv")

else:
    st.info("Please upload a CSV file to proceed.")
