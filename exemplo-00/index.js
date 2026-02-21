import tf, { input } from '@tensorflow/tfjs-node'

async function trainModel(inputXs, outputYs) { 
    const model = tf.sequential();
    // Primeira camada da rede:
    // Recebe uma entrada de 7 posições (idade normalizada + 3 cores + 3 localizações )

    // 80 neuronios = aqui coloquei tudo isso, pq tem pouca base de treino
    // quanto mais neuronios, mais complexidade a rede pode aprender
    // e consequentemente, mais processamento ela vai usar

    // A ReLU age como um filtro:
    // É como se ela deixasse somente os dados interessantes seguirem viagem na rede
    /// Se a informação chegou nesse neuronio é positiva, passa para frente!
    // se for zero ou negativa, pode jogar fora, nao vai servir para nada
    model.add(
        tf.layers.dense({ 
            inputShape: [7], // Recebe uma entrada de 7 posições (idade normalizada + 3 cores + 3 localizações )
            units: 80,  // Neurónios usados no teste
            activation: 'relu'  // A ReLU age como um filtro:
         }
    ))


    // Saida: 3 neuronios
    // um para cada categoria (premium, medium, basic)
    model.add(tf.layers.dense({ 
        units: 3, // neuronios
        activation: 'softmax' // activation: softmax normaliza a saida em probabilidades
    }))

    // Compilando o modelo
    // optimizer Adam ( Adaptive Moment Estimation)
    // é um treinador pessoal moderno para redes neurais:
    // ajusta os pesos de forma eficiente e inteligente
    // aprender com historico de erros e acertos

    // loss: categoricalCrossentropy
    // Ele compara o que o modelo "acha" (os scores de cada categoria)
    // com a resposta certa
    // a categoria premium será sempre [1, 0, 0]

    // quanto mais distante da previsão do modelo da resposta correta
    // maior o erro (loss)
    // Exemplo classico: classificação de imagens, recomendação, categorização de
    // usuário
    // qualquer coisa em que a resposta certa é "apenas uma entre várias possíveis"
    model.compile({ 
        optimizer: 'adam', // (Adaptive moment estimation),
        loss: 'categoricalCrossentropy', // 
        metrics: ['accuracy'] // 
    })


    // Treinamento do modelo
    // verbose: desabilita o log interno (e usa só callback)
    // epochs: quantidade de veses que vai rodar no dataset
    // shuffle: embaralha os dados, para evitar viés
    await model.fit(
        inputXs,
        outputYs,
        { 
            verbose: 0,
            epochs: 100,
            shuffle: true, 
            callbacks: { 
                onEpochEnd: (epoch, log) => { 
                    console.log( `Epoch: ${epoch}: loss = ${log.loss}` )
                } 
            }
        }
    )




}



const idade_normalizada = (min, max, idade) => { 
    return Number((idade - min ) / (max - min).toFixed(2))
}


// Exemplo de pessoas para treino (cada pessoa com idade, cor e localização)
// const pessoas = [
//     { nome: "Erick", idade: 30, cor: "azul", localizacao: "São Paulo" },
//     { nome: "Ana", idade: 25, cor: "vermelho", localizacao: "Rio" },
//     { nome: "Carlos", idade: 40, cor: "verde", localizacao: "Curitiba" },
//     { nome: "Jorge", idade: 33, cor: "verde", localizacao: "São Paulo" }
// ];

// Vetores de entrada com valores já normalizados e one-hot encoded
// Ordem: [idade_normalizada, azul, vermelho, verde, São Paulo, Rio, Curitiba]
// const tensorPessoas = [
//     [0.33, 1, 0, 0, 1, 0, 0], // Erick
//     [0, 0, 1, 0, 0, 1, 0],    // Ana
//     [1, 0, 0, 1, 0, 0, 1]     // Carlos
// ]

console.log('\n\n\n\n\n\n\n')
console.log(idade_normalizada(25, 40, 33))
console.log('\n\n\n\n\n\n\n')

// Usamos apenas os dados numéricos, como a rede neural só entende números.
// tensorPessoasNormalizado corresponde ao dataset de entrada do modelo.
const tensorPessoasNormalizado = [
    [0.33, 1, 0, 0, 1, 0, 0], // Erick
    [0, 0, 1, 0, 0, 1, 0],    // Ana
    [1, 0, 0, 1, 0, 0, 1],     // Carlos
    [ idade_normalizada(25, 40, 33), 0, 0, 1, 0, 0, 1]     // Jorge
]

// Labels das categorias a serem previstas (one-hot encoded)
// [premium, medium, basic]
const labelsNomes = ["premium", "medium", "basic"]; // Ordem dos labels
const tensorLabels = [
    [1, 0, 0], // premium - Erick
    [0, 1, 0], // medium - Ana
    [0, 0, 1],  // basic - Carlos
    [0, 0, 1]  // basic - Carlos
];


// Criamos tensores de entrada (xs) e saída (ys) para treinar o modelo
const inputXs = tf.tensor2d(tensorPessoasNormalizado)
const outputYs = tf.tensor2d(tensorLabels)


inputXs.print();
outputYs.print();


// Quantos mais dados, melhor
const model = trainModel(inputXs, outputYs)