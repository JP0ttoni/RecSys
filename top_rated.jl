using CSV
using DataFrames
using Statistics

movies = CSV.read("ml-latest-small/movies.csv", DataFrame)
ratings = CSV.read("ml-latest-small/ratings.csv", DataFrame)
len_movies = length(movies[:,1])
len_ratings = length(ratings[:,1])

ranking = zeros(Float64, len_movies, 4)
ranking[:,1] = movies[:,1]
media_geral = mean(ratings[:,3])
k = 30

for i=1:len_movies
    pos =  findall(x -> x == movies[i,1], ratings[:,2])
    soma = 0
    for j=1:length(pos)
        soma = soma + ratings[pos[j], 3]
    end
    if soma <= 0
        continue
    end
    ranking[i,2] = (soma + k * media_geral)/(length(pos) + k)  
    ranking[i,3] = soma/length(pos)
    ranking[i,4] = length(pos)
end

ranking = round.(ranking[sortperm(ranking[:,2], rev = true), :], digits=1)

df = DataFrame(ranking, [:id, :classificação, :nota, :avaliações])

CSV.write("ranking_filmes.csv", df)

ranking