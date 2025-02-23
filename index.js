// index.js
const path = require('path')
const { LogisticRegression } = require(path.join(__dirname, './build/Release/logistic_regression_classifier.node'));

module.exports = { LogisticRegression };
