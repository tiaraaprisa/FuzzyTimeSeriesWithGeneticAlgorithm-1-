from reguler_fuzzy import ChengReguler
import pygad


# Kelas Cheng GA yang dipanggil di main
class ChengGA():
    def __init__(self, param):
        # Mendapatkan data
        self.data = param["data"]

        # Mendapatkan parameter
        self.num_generations = param["num_generations"]
        self.num_solutions = param["num_chromosome"]
        self.mutation_rate = param["mutation"]
        self.crossover_rate = param["crossover"]
        # self.update_graph = param["graph"]
        self.num_gens = 6
        self.num_parents_mating = 3

        self.history = []


    # Callback function yang dipanggil untuk setiap start generasi
    def callback_start(self, ga_instance):
        # Iterate seluruh populasi (solusi)
        for i, s in enumerate(ga_instance.population):

            # Sort dari terkecil ke yang terbesar
            s.sort()
            # Assign nilai yang telah di sort
            ga_instance.population[i] = s


    # Callback function yang dipanggil untuk setiap setelah generasi

    def callback_generation(self, ga_instance):

        last_fitness = ga_instance.best_solution()[1]
        self.history.append(1 / last_fitness)
        # self.update_graph.line_chart(self.history)

    # FUngsi untuk melakukan prediksi dengan cheng
    def forecast(self, is_ga, gen):

        #  Membuat class cheng dimana gen = True
        cheng = ChengReguler({
            "data": self.data,
            "ga":is_ga,
            "gen": gen
          })
        # Mengembalikan kelas cheng
        return cheng

    # Callback fitness function
    def fitness_func(self, ga_instance, solution, solution_idx):

        # Melakukan prediksi dan mendapatkan kelas cheng yang telah di training
        result = self.forecast(True, solution)

        # Mengembalikan fitness score dimanaa 1/mape, jika error makin kecil maka fitness semakin besar
        # fitness semakin besar maka GA akan menggunakan solusi tersebut
        return 1 / result.mape

    def run(self):

        # Membuat GA Instance
        ga_instance = pygad.GA(num_generations=self.num_generations,
                               num_parents_mating=self.num_parents_mating,
                               fitness_func=self.fitness_func,
                               sol_per_pop=self.num_solutions,
                               num_genes=self.num_gens,
                               mutation_probability=self.mutation_rate,
                               crossover_probability= self.crossover_rate,
                               crossover_type="two_points",
                               on_generation=self.callback_generation,
                               on_start=self.callback_start)

        # Inisialisasi Populasi
        ga_instance.initialize_population(
            low=min(self.data),
            high=max(self.data) ,
            allow_duplicate_genes=False,
            mutation_by_replacement=False,
            gene_type=int
        )
        # print(ga_instance.population)

        # Training
        ga_instance.run()

        # Melakukan prediksi dengan solusi terbaik dari GA
        best_cheng =  self.forecast(True, ga_instance.best_solution()[0])

        # Mengembalikan data prediksi dan data error
        return best_cheng.forecast_result, best_cheng.forecast_error


