m = [5 10 -5 22; 1 33 15 3; 8 29 12 1; 3 11 39 20]

maiores = sort(m[:], rev = true)[1:3]#está pegando elementos do 1 ao 3 m[:] transforma em vetor

for i=1:length(maiores)
    indice = findall(isequal(maiores[i]), m)
    m[indice] .= 0 #zeros(Int, length(indice))
    println(m)
end

# criar um dataset rand

#id_user, id_item, nota

users = 943
items = 1648
total = 10

dataset = zeros(Int,total, 3)

for i=1:total
    dataset[i,1] = round.(( rand() .* users - 1) .+ 1)
    dataset[i,2] = round.((rand() .* items - 1) .+ 1)
    dataset[i,3] = round.((rand() .* 4) .+ 1)
end

dataset