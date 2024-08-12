import numpy as np

# Kelas Cheng Reguler yang dipanggi di kelas main
class ChengReguler:
    def __init__(self, param):

        # Mendapatkan Data
        self.data = param["data"]
        self.use_ga = param["ga"]
        self.gen = param["gen"]

        # Melakukan Training dan Forecast
        self.forecast()

        # Melakukan Perhitungan Error
        self.error()

    def forecast(self):

        # Menghitung initial Parameter untuk cheng
        data = self.data
        gen = self.gen
        U = [min(data), max(data)]
        range_ = max(data) - min(data)
        n = len(data)
        k = 7
        i = range_ / k
        average = n / k

        # Jika Menggunakan algoritma genetik, maka gunakan gen dari algoritma tersebut
        # Karena dari fungsi ChengGA sudah mengurutkan gen dari terkecil ke terbesar, maka langsung buat interval saja
        if self.use_ga:
            interval = [[min(data), gen[0]]]
            for count in range(1, k - 1):
                interval.append([gen[count - 1], gen[count]])

            interval.append([interval[len(interval)-1][1], max(data)])
        else:
            # Jika tidak menggunakan GA, bagi rata panjang data menjadi 7 kelas
            interval = [[min(data) + i * count, min(data) + i * (count + 1)] for count in range(k)]

        occurance = [0] * int(k)

        # Menghitung jumlah data yang masuk ke setiap kelas
        for value in data:
            for key2, interval_u in enumerate(interval):
                if interval_u[0] <= value <= interval_u[1]:
                    occurance[key2] += 1
                    break

        self.interval_awal = interval
        self.occur_awal = occurance

        # Block kode yang dieksekusi ketika masih ada jumlah data pada setiap kelas yang melebihi nilai rata - rata
        keep_divide = True
        while(keep_divide):

            # Iterate setiap Interval, pecah menjadi dua bagian
            for i in range(len(interval)):

                # Jika jumlah data lebih dari average, maka split interval
                if occurance[i] >= average:
                    bawah = interval[i][0]
                    atas = interval[i][1]
                    mid = (atas - bawah) / 2

                    # Splitting interval bagian pertama dari batas bawah ke nilai mid interval
                    insert1 = [bawah, bawah + mid]
                    insert1_occurance = 0

                    # Splitting interval bagian kedua dari mid interval ke batas atas
                    insert2 = [bawah + mid, atas]
                    insert2_occurance = 0


                    # Insert Interval baru bagian bawah
                    interval[i] = insert1
                    count = i + 1

                    # Insert interval baru bagian atas
                    # Count:Count untuk mendorong index setelahnya
                    interval[count:count] = [insert2]

                    # Set Occurance baru bagian bawah
                    occurance[i] = insert1_occurance
                    count_occur = i + 1

                    # set occurance baru bagian atas
                    # count_occur:count_occur untuk mendorong index setelahnya
                    occurance[count_occur:count_occur] = [insert2_occurance]

            occurance = [0] * len(interval)

            # Kembali hitung occurance setiap interval
            for value in data:
                for key2, interval_u in enumerate(interval):
                    if interval_u[0] <= value <= interval_u[1]:
                        occurance[key2] += 1
                        break
            keep_divide = False

            # check jumlah datasetiap interval, jika masih ada yang lebih dari rata -rata, maka lakukan kembali splitting

            for occur in occurance:
                if occur > average:
                    keep_divide = True



        # Membuat Fuzifikasi setiap data
        A = []
        for value in data:
            for key2, interval_u in enumerate(interval):
                if value >= interval_u[0] and value <= interval_u[1]:
                    A.append(key2)
                    break

        # Membuat Fuzzy Logical Relation Group yang segera di convert menjadi matriks berdasarkan FLRG
        matriks = [[0] * len(interval) for _ in range(len(interval))]
        for i in range(1, len(data)):
            matriks[A[i - 1]][A[i]] += 1


        # Membuat matriks kejadian menjadi matriks bobot
        matriks_bobot = [[round(value / sum(row), 2) if sum(row) > 0 else 0 for value in row] for row in matriks]

        # Melakukan Forecasting
        mid = [(value[0] + value[1]) / 2 for value in interval]
        forecast = [sum(mid[j] * matriks_bobot[A[i]][j] for j in range(len(mid))) for i in range(len(A))]

        # Melakukan Adaptive Forcasting
        adaptif = [data[i] + 1 * (forecast[i] - data[i]) for i in range(len(data))]

        # Simpan Semua data
        self.forecast = forecast
        self.interval_akhir = interval
        self.occur_akhir = occurance
        self.A = A
        self.matriks = matriks
        self.matriks_bobot = matriks_bobot
        self.forecast_result = adaptif

        self.min = min(data)
        self.max = max(data)
        self.range = range_
        self.I = i
        self.n = n
        self.K = k
        self.average = average


    def error(self):
        # Mendapatkan data
        actual = np.array(self.data)

        # Mendapatkan data hasil prediksi
        forecasted = np.array(self.forecast_result)

        # Mendapatkan error setiap data
        self.forecast_error = actual - forecasted

        # Mendapatkan mape
        mape = np.mean(np.abs((actual - forecasted) / actual)) * 100
        self.first10mape = np.abs((actual - forecasted) / actual)
        self.mape = mape