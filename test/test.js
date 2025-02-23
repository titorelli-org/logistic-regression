const fs = require('fs')
const path = require('path')
const { LogisticRegression } = require('../build/Release/logistic_regression_classifier.node')
const { PorterStemmerRu } = require('natural')

const loadDataset = (fileName) => {
    return fs.readFileSync(path.join(__dirname, fileName), 'utf-8')
        .trim()
        .split('\n')
        .map(text => JSON.parse(text.trim()))
        .map(item => ({
            text: PorterStemmerRu.tokenizeAndStem(item.text).join(' '),
            label: item['спам'] ? 'spam' : 'ham'
        }))
}

const smallBenchmark = () => {
    let trainExamples = loadDataset('train-small-dataset.ljson')
    let testExamples = loadDataset('test-small-dataset.ljson')

    return { trainExamples, testExamples }
}

const fullBenchmark = () => {
    const fullDataset = loadDataset('yandex-dataset.json')

    let trainExamples = []
    let testExamples = []

    {
        const hamExamples = fullDataset.filter(({ label }) => label === 'ham').slice(0, 9127)
        const spamExamples = fullDataset.filter(({ label }) => label === 'spam').slice(0, 99)

        trainExamples = [...spamExamples, ...hamExamples]
    }

    {
        const hamExamples = fullDataset.filter(({ label }) => label === 'ham').slice(9128)
        const spamExamples = fullDataset.filter(({ label }) => label === 'spam').slice(100)

        testExamples = [...spamExamples, ...hamExamples]
    }

    return { trainExamples, testExamples }
}

const makeSmallDataset = () => {
    const { trainExamples, testExamples } = fullBenchmark()

    const stringifyExample = (text, label) => {
        const result = { text }

        if (label === 'spam') {
            result['спам'] = 1
        }

        if (label === 'ham') {
            result['не спам'] = 1
        }

        return JSON.stringify(result)
    }

    fs.writeFileSync(path.join(__dirname, 'train-small-dataset.ljson'), '', 'utf-8')
    fs.writeFileSync(path.join(__dirname, 'test-small-dataset.ljson'), '', 'utf-8')

    for (const { text, label } of trainExamples) {
        fs.appendFileSync(
            path.join(__dirname, 'train-small-dataset.ljson'),
            stringifyExample(text, label) + '\n',
            'utf-8'
        )
    }

    for (const { text, label } of testExamples) {
        fs.appendFileSync(
            path.join(__dirname, 'test-small-dataset.ljson'),
            stringifyExample(text, label) + '\n',
            'utf-8'
        )
    }
}

let trainExamples = [], testExamples = []

switch (process.argv[2]) {
    default:
    case 'small':
        ({ trainExamples, testExamples } = smallBenchmark())
        break
    case 'full':
        ({ trainExamples, testExamples } = fullBenchmark())
        break
    case 'make':
        makeSmallDataset()
        break
}

if (process.argv[2] === 'make')
    process.exit(0)

let nativeClassifier = new LogisticRegression({ learningRate: 0.01, iterations: 1000 })
// const jsClassifier = new LogisticRegressionClassifier(PorterStemmerRu)

if (fs.existsSync(path.join(__dirname, 'model'))) {
    console.time('load native')

    nativeClassifier.loadModel(path.join(__dirname, 'model'))

    console.timeEnd('load native')
} else {
    const trainTexts = trainExamples.map(({ text }) => text)
    const trainLabels = trainExamples.map(({ label }) => label === 'ham' ? 0 : 1)

    console.time('train native')

    // for (let i = 0; i < trainExamples.length; i++) {
    //     const { text, label } = trainExamples[i]

    //     nativeClassifier.train([text], [label === 'ham' ? 0 : 1])
    // }

    nativeClassifier.train(
        trainTexts,
        trainLabels
    )

    console.timeEnd('train native')

    // console.time('memory test')
    
    // for (let i=0; i<10; i++) {
    //     console.log(`--${i}--`)

    //     const c = new LogisticRegression({ learningRate: 0.01, iterations: 1000 })
        
    //     c.train(
    //         trainTexts,
    //         trainLabels
    //     )
    // }

    // console.timeEnd('memory test')

    console.time('saving native')

    nativeClassifier.saveModel(path.join(__dirname, 'model'))

    console.timeEnd('saving native')
}

nativeClassifier = new LogisticRegression({ learningRate: 0.01, iterations: 1000 })

console.time('loading native')

nativeClassifier.loadModel(path.join(__dirname, 'model'))

console.timeEnd('loading native')

// console.time('train js')

// for (const { text, label } of trainExamples) {
//     jsClassifier.addDocument(text, label)
// }

// jsClassifier.train()

// console.timeEnd('train js')

let errorsCount = 0

console.time('benchmark')

for (const { label, text } of testExamples) {
    const nativeResult = nativeClassifier.classify(text)
    const nativeClassifierLabel = nativeResult <= 0.5 ? 'ham' : 'spam'

    if (label !== nativeClassifierLabel) {
        errorsCount += 1
    }

    // const jsClassifierLabel = jsClassifier.classify(text)

    // if (nativeClassifierLabel !== jsClassifierLabel) {
    //     errorsCount += 1
    // }
}

console.timeEnd('benchmark')

const errorRate = errorsCount / testExamples.length

console.log('Error rate:', errorRate)
