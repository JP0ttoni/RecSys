using Statistics

m = [5 10 -5 22; 1 33 15 3; 8 29 12 1; 3 11 39 20]


function normalize(matriz)
    matriz = float.(matriz)
    for i=1 : size(matriz,2)#retorna o número de colunas
        col = matriz[:,i]
        for j=1 : length(col)
            col[j] = (col[j] - mean(matriz[:,i]))/std(matriz[:,i])
        end
        matriz[:,i] = col
    end
    println(round.(matriz, digits=4))
end

normalize(m)