entrada = [1 2 3; 4 7 6]

(linhas, colunas) = size(entrada)

saida = zeros(Int, linhas, colunas)

for i=1:linhas
    vetor = entrada[i,:] #retorna 123
    saida[i,:] = maximum(vetor) .== entrada[i,:] #o ponto é pra comparar de ponta a ponta do vetor
    println(saida)
end
