using CSV
using DataFrames

movies = CSV.read("ml-latest-small/movies.csv", DataFrame)
ratings = CSV.read("ml-latest-small/ratings.csv", DataFrame)
len_movies = length(movies[:,1])
len_ratings = length(ratings[:,1])
userid = movies[1:10,2]

ranking = zeros(Int64, len_movies, 2)
ranking[:,1] = movies[:,1]

for i=1:len_ratings
    pos = findall(x -> x == ratings[i,2], ranking[:,1])
    ranking[pos,2]  = ranking[pos,2] .+ 1
end

ranking = ranking[sortperm(ranking[:,2], rev = true), :][1:10,:]

ranking