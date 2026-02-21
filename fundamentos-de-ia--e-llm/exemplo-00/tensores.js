import tf from '@tensorflow/tfjs-node'

/**
 * Tensores
 * Vetores ou Listas
 */

/**
 * Função que vai criar o modelo de treinamento
 * @param {*} tensor_de_entrada | Dados que vão ser inseridos para o treinamento
 * @param {*} tensor_de_saida | Dados que vão ser previstos
 * @returns | Modelo de treinamento
 */
const modelo_de_treinamento = async (tensor_de_entrada, tensor_de_saida) => {
    const model = tf.sequential(); // Estamos instanciando o modelo de treinamento



    /**
     * Essa parte de redes neurais é a mais importante.
     * Fica explicada de forma mais didática no arquivo ./assets/exemplo-rede-detalhada.png
     */

    /**
     * ======= Camada Oculta =======
     * 
     * A camada oculta é a camada que vai processar os dados de entrada e produzir uma saída.
     * 
     * A primeira camada deve ter como entrada 7 posições (no caso do exemplo, a idade normalizada + 3 cores + 3 localizações )
     * Cada posição vai representar uma coluna da matriz de entrada.
     * 
     * O parametro inputShape é o numero de colunas da matriz de entrada.
     * 
     * O parametro units é o numero de neurónios na camada oculta.
     * 
     * O parametro activation é a função de ativação da camada oculta.
     * 
     * A ReLU age como um filtro:
     * É como se ela deixasse somente os dados interessantes seguirem viagem na rede
     * 
     * ReLu = Rectified Linear Unit
     */
    
    model.add(tf.layers.dense({
        inputShape: [7], 
        // Aumentei bastante os neuronios para ver se o modelo consegue aprender melhor. porque com 80 ele não conseguia sair de 0.50 de acerto.
        units: 3000, // 3000 neurónios usados no teste (Quanto mais neuronios, mais complexidade a rede pode aprender)
        activation: 'relu'
    }));


    /**
     * ======= Camada de Saída =======
     * 
     * A camada de saída é a camada que vai produzir a saída final.
     * 
     * A camada de saída deve ter como entrada o numero de neurónios da camada oculta.
     * 
     * O parametro units é o numero de neurónios na camada de saída.
     * 
     * O parametro activation é a função de ativação da camada de saída.
     */
    model.add(tf.layers.dense({
        units: 3, // 3 neurónios usados no teste (Quanto mais neuronios, mais complexidade a rede pode aprender)
        activation: 'softmax'
    }));


    /**
     * ======= Compilando o Modelo =======
     * 
     * O parametro optimizer é o algoritmo de otimização usado para treinar o modelo.
     * Usamos o adam (Adaptive moment estimation), pois é um algoritmo de otimização moderno e eficiente que ajusta os pesos de forma eficiente e inteligente
     * 
     * O parametro loss é a função de erro usada para treinar o modelo.
     * Usamos o categoricalCrossentropy, pois é uma função de erro que compara o que o modelo "acha" (os scores de cada categoria)
     * com a resposta certa
     * a categoria premium será sempre [1, 0, 0]
     * a categoria gold será sempre [0, 1, 0]
     * a categoria basic será sempre [0, 0, 1]
     * 
     * Exemplo classico: classificação de imagens, recomendação, categorização de usuário
     * qualquer coisa em que a resposta certa é "apenas uma entre várias possíveis"
     * 
     * 
     * O parametro metrics é o array de métricas usadas para avaliar o modelo.
     * Usamos o accuracy, pois é uma métrica que mede a precisão do modelo.
     * 
     * A precisão é a quantidade de acertos dividido pelo total de dados.
     * 
     * Exemplo:
     * Se o modelo acertou 100 dados e errou 0 dados, a precisão é 100%.
     * Se o modelo acertou 90 dados e errou 10 dados, a precisão é 90%.
     * 
     * A precisão é uma métrica que mede a precisão do modelo.
     */
    model.compile({
        optimizer: 'adam',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });


    /**
     * ======= Treinando o Modelo =======
     * 
     * O parametro epochs é a quantidade de veses que vai rodar no dataset.
     * 
     * O parametro shuffle é um booleano que indica se vai embaralhar os dados a cada vez que o modelo vai rodar.
     * 
     * O parametro verbose é um booleano que indica se vai mostrar o log do treinamento.
     * 
     * O parametro callbacks é um array de callbacks que vai ser executado a cada época.
     */
    await model.fit(
        tensor_de_entrada,
        tensor_de_saida,
        {
            verbose: 0, // Desabilita o log interno (e usa só callback)
            /**
             * O parametro epochs é a quantidade de veses que vai rodar no dataset.
             * 
             * 100 é um valor arbitrário, pode ser alterado para o que você achar melhor
             * 
             * Exemplo:
             * Se o modelo treinar com 100 épocas, ele vai treinar com os mesmos dados 100 vezes.
             * Isso é importante para evitar viés.
             */
            epochs: 100,
            /**
             * O parametro shuffle é um booleano que indica se vai embaralhar os dados a cada vez que o modelo vai rodar.
             * 
             * Embaralhar os dados é importante para evitar viés.
             * Se não embaralhar, o modelo vai treinar com os mesmos dados a cada vez que rodar.
             * Isso é importante para evitar viés.
             * 
             * Exemplo:
             * Se o modelo treinar com os mesmos dados a cada vez que rodar, ele vai se adaptar aos dados e não vai generalizar bem.
             * Isso é importante para evitar viés.
             */
            shuffle: true,
            callbacks: {
                onEpochEnd: (epoch, log) => {
                    console.clear();
                    console.log( `Epoch: ${epoch}: loss = ${log.loss}` )
                }
            }
        }
    );


    return model;
}


/** 
 * Essa função vai gerar um valor para a idade entre 1 e 0 
 * onde 1 é a idade máxima e 0 é a idade minima.
 */
const fnNormizaIdade = (idade_min, idade_max) => (idade) => { 
    return (idade - idade_min) / (idade_max - idade_min ) 
}


const IDADE_MIN = 22;
const IDADE_MAX = 41


const pessoas = [
    { 
        nome: 'Jorge', 
        idade: 33,
        cidade: 'São Paulo'
    },
    {
        nome: 'Ana',
        idade: 27,
        cidade: 'Rio de Janeiro'
    },
    {
        nome: 'Carlos',
        idade: 41,
        cidade: 'Belo Horizonte'
    },
    {
        nome: 'Beatriz',
        idade: 22,
        cidade: 'Porto Alegre'
    },
    {
        nome: 'Marina',
        idade: 35,
        cidade: 'Recife'
    }
]

/**
 * Da uma olhada no .csv para entender um pouco melhor o que está acontecendo.
 * 
 * O tensor está assim: 
 * [idade_normalizada, azul, vermelho, verde, São Paulo, Rio, Curitiba]
 */
const tensor_normalizado_pessoas = [
    [fnNormizaIdade(IDADE_MIN, IDADE_MAX)(33),1,0,0,1,0,0] // Jorge
    ,[fnNormizaIdade(IDADE_MIN, IDADE_MAX)(27),1,0,0,1,0,0] // Ana 
    ,[fnNormizaIdade(IDADE_MIN, IDADE_MAX)(41),0,1,0,0,1,0] // Carlos 
    ,[fnNormizaIdade(IDADE_MIN, IDADE_MAX)(22),0,0,1,0,0,1] // Beatriz
    ,[fnNormizaIdade(IDADE_MIN, IDADE_MAX)(35),0,0,1,0,0,1] // Marina
]

// Aqui temos as categorias disponíveis para serem previstas.
const categorias = [
    "Premium",
    "Gold",
    "Basic"
]

/***
 *  Essas são as categorias de cada uma das pessoas.
 *  Jorge   → Premium
    Ana     → Gold
    Carlos  → Basic
    Beatriz → Premium
    Marina  → Gold
 */

/**
 * O tensor está assim: 
 * [Premium, Gold, Basic]
 */
const tensor_categorias = [
    [1, 0, 0], // Jorge
    [0, 1, 0], // Ana
    [0, 0, 1], // Carlos
    [1, 0, 0], // Beatriz
    [0, 1, 0], // Marina
]


// Agora precisamos fazer uma matriz com as informações das pessoas
// A Matriz precisa atribuir distribuir entre 0 e 1 as informações


const tensor_de_entrada = tf.tensor2d(tensor_normalizado_pessoas)
const tensor_de_saida = tf.tensor2d(tensor_categorias);


tensor_de_entrada.print();
tensor_de_saida.print();


console.log('\n\n\n\n\n\n\n')

const model = modelo_de_treinamento(
    tensor_de_entrada, // Dados que vão ser inseridos para o treinamento
    tensor_de_saida // 
);