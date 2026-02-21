import tf from '@tensorflow/tfjs-node'
import fs from 'fs';

import { fnNormizaIdade, fnOneHotEncode, LABEL_PLANOS, normalizaDadosParaTensor, PLANOS } from './helpers/utils.tools.js';

/**
 * Tensores
 * Vetores ou Listas
 */

/**
 * Função que vai criar o modelo de treinamento
 * @param {tf.Tensor2D} tensor_de_entrada | Dados que vão ser inseridos para o treinamento
 * @param {tf.Tensor2D} tensor_de_saida | Dados que vão ser previstos
 * @param {Object} configs | Configurações do modelo
 * @param {number} configs.tamanho_colunas_tensor_de_entrada | Tamanho das colunas do tensor de entrada
 * @returns | Modelo de treinamento
 */
const modelo_de_treinamento = async (tensor_de_entrada, tensor_de_saida, configs) => {
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
        inputShape: [configs.tamanho_colunas_tensor_de_entrada], 
        // Aumentei bastante os neuronios para ver se o modelo consegue aprender melhor. porque com 80 ele não conseguia sair de 0.50 de acerto.
        units: 80, // 3000 neurónios usados no teste (Quanto mais neuronios, mais complexidade a rede pode aprender)
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

    let lastLoss = 0;
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
                    // console.log( `Epoch: ${epoch}: loss = ${log.loss}` )
                    lastLoss = log.loss;
                }
            }
        }
    );

    console.log('Modelo treinado com sucesso! Loss:', lastLoss)

    return model;
}

/**
 * Função que faz a predição de resultado
 * Com ela usamos o modelo que treinamos para fazer a predição do resultado, passando o dado que queremos analisar.
 * @param {tf.Model} model | Modelo treinado
 * @param {tf.Tensor2D} tensor_de_entrada_para_analisar | Aqui passamos o dado que queremos analisar
 * @returns {Promise<Array>} | Resultado da predição
 */
const predicao_de_resultado = async (model, tensor_de_entrada_para_analisar) => {
    const predicao = await model.predict(tensor_de_entrada_para_analisar);
    const predicaoArray = await predicao.array();

    console.log('\n\n')
    console.log('Resultado da predição:', predicaoArray)
    console.log('\n\n')


    return predicaoArray[0].map((prob, index) => ({ prob: prob * 100, index }));
}





const PESSOAS_TREINAMENTO = fs.readFileSync('./assets/pessoas.json', 'utf8');
const pessoasJson = JSON.parse(PESSOAS_TREINAMENTO);




/**
 * Da uma olhada no .csv para entender um pouco melhor o que está acontecendo.
 * 
 * O tensor está assim: 
 * [idade_normalizada, azul, vermelho, verde, São Paulo, Rio, Belo Horizonte, Porto Alegre, Recife]
 */
 // [idadeTensor(33),1,0,0,1,0,0] // Jorge
// ,[idadeTensor(27),1,0,0,1,0,0] // Ana 
// ,[idadeTensor(41),0,1,0,0,1,0] // Carlos 
// ,[idadeTensor(22),0,0,1,0,0,1] // Beatriz
// ,[idadeTensor(35),0,0,1,0,0,1] // Marina
const tensor_normalizado_pessoas = pessoasJson.map((pessoa) => normalizaDadosParaTensor(pessoa));


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
*   [1, 0, 0], // Jorge
    [0, 1, 0], // Ana
    [0, 0, 1], // Carlos
    [1, 0, 0], // Beatriz
    [0, 1, 0], // Marina
 */
const tensor_categorias = pessoasJson.map((pessoa) => {
    return [
        fnOneHotEncode(pessoa.plano, PLANOS.PREMIUM), // Premium
        fnOneHotEncode(pessoa.plano, PLANOS.GOLD), // Gold
        fnOneHotEncode(pessoa.plano, PLANOS.BASIC), // Basic
    ]
});


// Agora precisamos fazer uma matriz com as informações das pessoas
// A Matriz precisa atribuir distribuir entre 0 e 1 as informações


const tensor_de_entrada = tf.tensor2d(tensor_normalizado_pessoas)
const tensor_de_saida = tf.tensor2d(tensor_categorias);

console.log('\n\n\n\n')
console.log('Tensor de entrada:')
tensor_de_entrada.print();

console.log('\n\n\n\n')
console.log('Tensor de saída:')
tensor_de_saida.print();


console.log('\n\n\n')
console.log('Treinando o modelo...')

const model = await modelo_de_treinamento(
    tensor_de_entrada, // Dados que vão ser inseridos para o treinamento
    tensor_de_saida, // Dados que vão ser previstos
    { 
        // Vamos informar de forma dinamica o shape do tensor de entrada
        tamanho_colunas_tensor_de_entrada: tensor_normalizado_pessoas[0].length
    }
);






const nova_pessoa = {
    idade: 33,
    cor: 'Azul',
    cidade: 'São Paulo',
    plano: 'Premium'
}

const entrada_para_analisar = [ normalizaDadosParaTensor(nova_pessoa) ]

const tensor_de_entrada_para_analisar = tf.tensor2d(entrada_para_analisar);

console.log('\n\n\n\n')
console.log('Tensor de entrada para analisar:')
tensor_de_entrada_para_analisar.print();


console.log('\n\n')
console.log('Fazendo a predição...')

const resultado = await predicao_de_resultado(model, tensor_de_entrada_para_analisar);

console.log('\n\n')
console.log('Resultado da predição:', resultado);
console.log('\n\n')

/**
 * Mapeia pelo maior ( ordenando pelo que tem a maior probabilidade)
 */
const results = resultado
    .sort((a, b) => b.prob - a.prob)
    .map(({ prob, index }) => `${LABEL_PLANOS[index]}: (${prob.toFixed(2)}%)`)
    .join('\n');

console.log('\n\n')
console.log('Resultado da predição:', results);
console.log('\n\n')