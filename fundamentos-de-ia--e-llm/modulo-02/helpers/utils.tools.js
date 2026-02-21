/**
 * Função que normaliza a idade entre 0 e 1
 * Sendo que 1 é a idade máxima e 0 é a idade mínima.
 * @param {*} idade_min | Idade mínima
 * @param {*} idade_max | Idade máxima
 * @returns | Valor normalizado da idade
 */
export const fnNormizaIdade = (idade_min, idade_max) => (idade) => { 
    return Number((idade - idade_min ) / (idade_max - idade_min).toFixed(2))
}

/**
 * Função que faz o one-hot encode das categorias
 * Sendo que 1 é se der match com a categoria, e 0 caso contrario.
 * @param {*} categorias | Categorias disponíveis
 * @returns | Array de 0 e 1, onde 1 é se der match com a categoria, e 0 caso contrario.
 */
export const fnOneHotEncode = (valor, referencia) => {
    return valor === referencia ? 1 : 0
}


export const IDADE_MIN = 22;
export const IDADE_MAX = 41

export const CIDADES = { 
    'SAO_PAULO': 'São Paulo',
    'RIO_DE_JANEIRO': 'Rio de Janeiro',
    'BELHORIZONTE': 'Belo Horizonte',
    'PORTO_ALEGRE': 'Porto Alegre',
    'RECIPE': 'Recife'
}

export const CORES_FAVORITAS =  { 
    'AZUL': 'Azul',
    'VERMELHO': 'Vermelho',
    'VERDE': 'Verde'
}


export const LABEL_PLANOS = [ 'Premium', 'Gold', 'Basic' ]

export const PLANOS = { 
    'PREMIUM': 'Premium',
    'GOLD': 'Gold',
    'BASIC': 'Basic'
}


/**
 * Função que normaliza os dados para o tensor
 * @param {Object} pessoa | Pessoa com os dados
 * @returns {Array} | Array com os dados normalizados para o formato do tensor
 */
export const idadeNormalizadaParaTensor = fnNormizaIdade(IDADE_MIN, IDADE_MAX);

export const normalizaDadosParaTensor = (pessoa) => { 
    
    return [
        idadeNormalizadaParaTensor(pessoa.idade),
        fnOneHotEncode(pessoa.cor, CORES_FAVORITAS.AZUL), // Azul
        fnOneHotEncode(pessoa.cor, CORES_FAVORITAS.VERMELHO), // Vermelho
        fnOneHotEncode(pessoa.cor, CORES_FAVORITAS.VERDE), // Verde
        fnOneHotEncode(pessoa.cidade, CIDADES.SAO_PAULO), // São Paulo
        fnOneHotEncode(pessoa.cidade, CIDADES.RIO_DE_JANEIRO), // Rio de Janeiro
        fnOneHotEncode(pessoa.cidade, CIDADES.BELHORIZONTE), // Belo Horizonte
        fnOneHotEncode(pessoa.cidade, CIDADES.PORTO_ALEGRE), // Porto Alegre
        fnOneHotEncode(pessoa.cidade, CIDADES.RECIPE), // Recife 
    ]
}