declare module '@titorelli/logistic-regression' {
  class LogisticRegression {
    constructor(conf?: { learningRate?: number, iterations?: number, numFeatures?: number })

    train(docs: string[], labels: number[]): void

    classify(doc: string): number

    loadModel(filename: string): void

    saveModel(filename: string): void
  }
}
