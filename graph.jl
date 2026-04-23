using CSV
using DataFrames
using Statistics
using Plots

movies = CSV.read("ml-latest-small/movies.csv", DataFrame)
ratings = CSV.read("ml-latest-small/ratings.csv", DataFrame)
len_movies = length(movies[:,1])
len_ratings = length(ratings[:,1])
users = maximum(ratings[:,1])
mat = zeros(Int64, users, 2)

for i=1:users
    qntd = findall(x -> x == i, ratings[:,1])
    mat[i, 1] = i
    mat[i, 2] = length(qntd)
end

mat = mat[sortperm(mat[:,2], rev=true), :]

#contagem = combine(groupby(ratings, :userId), nrow => :qntd)

# Ordena de forma decrescente
#sort!(contagem, :qntd, rev=true)

display(mat)

graph = plot(mat[:, 2], 
             title="Distribuição de Avaliações por Usuário",
             xlabel="Ranking de Usuários (Mais ativos -> Menos ativos)",
             ylabel="Quantidade de Avaliações",
             lw=2, 
             color=:red)

gui()      # Abre a janela externa do backend (GR)
display(graph) # Força o VS Code a mostrar na aba de Plots
readline()