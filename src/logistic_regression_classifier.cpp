#include <napi.h>
#include <vector>
#include <unordered_map>
#include <string>
#include <sstream>
#include <cmath>
#include <numeric>
#include <functional> // Для std::hash
#include <fstream>
#include <iostream> // Для отладки

class LogisticRegression : public Napi::ObjectWrap<LogisticRegression> {
public:
    static Napi::Object Init(Napi::Env env, Napi::Object exports);
    LogisticRegression(const Napi::CallbackInfo& info);

private:
    Napi::Value Train(const Napi::CallbackInfo& info);
    Napi::Value Classify(const Napi::CallbackInfo& info);
    Napi::Value LoadModel(const Napi::CallbackInfo& info);
    Napi::Value SaveModel(const Napi::CallbackInfo& info);
    double CalculateLoss(const std::vector<std::vector<double>>& inputs, const std::vector<int>& labels);
    std::vector<double> HashingVectorize(const std::string& text, size_t numFeatures);

    std::vector<double> weights;
    std::unordered_map<std::string, size_t> featureDict; // Словарь для хранения признаков
    double learningRate = 0.01;
    int iterations = 200000;
    size_t numFeatures = 100000; // Количество признаков

     double sigmoid(double z);
     std::vector<double> hypothesis(const std::vector<double>& theta, const std::vector<std::vector<double>>& observations);
     double cost(const std::vector<double>& theta, const std::vector<std::vector<double>>& examples, const std::vector<double>& classifications);
     std::vector<double> descendGradient(std::vector<double> theta, std::vector<std::vector<double>> examples, const std::vector<double>& classifications);
};

Napi::Object LogisticRegression::Init(Napi::Env env, Napi::Object exports) {
    Napi::Function func = DefineClass(env, "LogisticRegression", {
        InstanceMethod("train", &LogisticRegression::Train),
        InstanceMethod("classify", &LogisticRegression::Classify),
        InstanceMethod("loadModel", &LogisticRegression::LoadModel),
        InstanceMethod("saveModel", &LogisticRegression::SaveModel),
        });

    exports.Set(Napi::String::New(env, "LogisticRegression"), func);
    return exports;
}

LogisticRegression::LogisticRegression(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<LogisticRegression>(info) {
    if (info.Length() > 0 && info[0].IsObject()) {
        Napi::Object options = info[0].As<Napi::Object>();
        if (options.Has("learningRate")) {
            learningRate = options.Get("learningRate").As<Napi::Number>().DoubleValue();
        }
        if (options.Has("iterations")) {
            iterations = options.Get("iterations").As<Napi::Number>().Int32Value();
        }
        if (options.Has("numFeatures")) {
            numFeatures = options.Get("numFeatures").As<Napi::Number>().Uint32Value();
        }
    }
}

std::vector<double> LogisticRegression::HashingVectorize(const std::string& text, size_t numFeatures) {
    std::vector<double> vector(numFeatures, 0.0);
    std::hash<std::string> hasher;

    std::istringstream stream(text);
    std::string token;

    while (stream >> token) {
        size_t hashIndex = hasher(token) % numFeatures;
        if (featureDict.find(token) == featureDict.end()) {
            featureDict[token] = hashIndex; // Добавляем новый признак в словарь
        }
        vector[featureDict[token]] += 1.0; // Увеличиваем значение признака
    }

    return vector;
}

double LogisticRegression::sigmoid(double z) {
    if (z < -709) return 0.0; // избегаем переполнения
    if (z > 709) return 1.0;  // избегаем переполнения
    return 1.0 / (1.0 + std::exp(-z));
}

 std::vector<double> LogisticRegression::hypothesis(const std::vector<double>& theta, const std::vector<std::vector<double>>& observations) {
    std::vector<double> result(observations.size());
    for (size_t i = 0; i < observations.size(); ++i) {
        double dot_product = 0.0;
        for (size_t j = 0; j < theta.size(); ++j) {
            dot_product += theta[j] * observations[i][j];
        }
        result[i] = sigmoid(dot_product);
    }
    return result;
}

 double LogisticRegression::cost(const std::vector<double>& theta, const std::vector<std::vector<double>>& examples, const std::vector<double>& classifications) {
    std::vector<double> hypothesisResult = hypothesis(theta, examples);
    double cost_1 = 0.0;
    double cost_0 = 0.0;
    for (size_t i = 0; i < classifications.size(); ++i) {
        if (classifications[i] == 1) {
            cost_1 += classifications[i] * log(hypothesisResult[i]);
        }
        else {
            cost_0 += (1 - classifications[i]) * log(1 - hypothesisResult[i]);
        }
    }
    return -(cost_1 + cost_0) / examples.size();
}





 std::vector<double> LogisticRegression::descendGradient(std::vector<double> theta, std::vector<std::vector<double>> examples, const std::vector<double>& classifications) {
    int maxIt = 500 * examples.size();
    double learningRate = 3.0;
    bool learningRateFound = false;
    std::vector<double> last;
    double current = 0.0;

    // Добавляем смещение
    for (auto& example : examples) {
        example.insert(example.begin(), 1.0);
    }
    theta.insert(theta.begin(), 0.0);

    while (!learningRateFound && learningRate != 0) {
        int i = 0;
        last.clear();

        while (true) {
            std::vector<double> hypothesisResult = hypothesis(theta, examples);
            std::vector<double> gradient(theta.size(), 0.0);

            for (size_t j = 0; j < theta.size(); ++j) {
                for (size_t k = 0; k < examples.size(); ++k) {
                    gradient[j] += (hypothesisResult[k] - classifications[k]) * examples[k][j];
                }
                gradient[j] /= examples.size();
                gradient[j] *= learningRate;
            }

            for (size_t j = 0; j < theta.size(); ++j) {
                theta[j] -= gradient[j];
            }

            current = cost(theta, examples, classifications);

            if (!last.empty() && current < last.back()) {
                learningRateFound = true;
                break;
            }

            if (!last.empty() && abs(last.back() - current) < 0.0001) {
                break;
            }

            if (++i >= maxIt) {
                throw std::runtime_error("Unable to find minimum");
            }

            last.push_back(current);
        }

        learningRate /= 3.0;
    }

    // Удаляем смещение
    theta.erase(theta.begin());
    return theta;
}




Napi::Value LogisticRegression::Train(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 2 || !info[0].IsArray() || !info[1].IsArray()) {
        Napi::TypeError::New(env, "Expected two arrays").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    Napi::Array inputArray = info[0].As<Napi::Array>();
    Napi::Array labelArray = info[1].As<Napi::Array>();

    std::vector<std::vector<double>> inputs;
    std::vector<int> labels;

    // Преобразуем входные данные и метки в векторы
    for (size_t i = 0; i < inputArray.Length(); i++) {
        std::string doc = inputArray.Get(i).As<Napi::String>().Utf8Value();
        inputs.push_back(HashingVectorize(doc, numFeatures));
        labels.push_back(labelArray.Get(i).As<Napi::Number>().Int32Value());
    }

    // Инициализируем веса
    std::vector<double> theta(numFeatures, 0.0);

    // Преобразуем метки в вектор<double>
    std::vector<double> doubleLabels(labels.begin(), labels.end());

    // Запуск градиентного спуска
    theta = descendGradient(theta, inputs, doubleLabels);

    // Обновляем веса
    weights = theta;

    return env.Undefined();
}



double LogisticRegression::CalculateLoss(const std::vector<std::vector<double>>& inputs, const std::vector<int>& labels) {
    double loss = 0.0;
    const double epsilon = 1e-15; // Малое значение для предотвращения ошибок при логарифмировании

    for (size_t i = 0; i < inputs.size(); i++) {
        // Вычисление предсказания с защитой от переполнения
        double prediction = sigmoid(std::inner_product(inputs[i].begin(), inputs[i].end(), weights.begin(), 0.0));

        // Ограничиваем значения предсказания, чтобы избежать логарифмов от 0 или 1
        prediction = std::max(epsilon, std::min(1.0 - epsilon, prediction));
        double logPrediction = std::log(prediction);
        double logOneMinusPrediction = std::log(1.0 - prediction);

        // Логистическая функция потерь
        loss += -labels[i] * logPrediction - (1 - labels[i]) * logOneMinusPrediction;
    }

    // Возвращаем усредненный loss
    return loss / inputs.size();
}



Napi::Value LogisticRegression::Classify(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1 || !info[0].IsString()) {
        Napi::TypeError::New(env, "Expected a string").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    std::string input = info[0].As<Napi::String>().Utf8Value();
    std::vector<double> vector = HashingVectorize(input, numFeatures);

    double z = std::inner_product(vector.begin(), vector.end(), weights.begin(), 0.0);
    double probability = sigmoid(z);

    // Устанавливаем порог для классификации
    double threshold = 0.5;  // Можете изменить порог для улучшения результатов
    int predictedClass = (probability >= threshold) ? 1 : 0;

    // Возвращаем предсказанный класс (1 или 0)
    return Napi::Number::New(env, predictedClass);
}


Napi::Value LogisticRegression::SaveModel(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1 || !info[0].IsString()) {
        Napi::TypeError::New(env, "Expected a string").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    std::string filename = info[0].As<Napi::String>();
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        Napi::Error::New(env, "Unable to open file").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    // Сохраняем веса
    size_t weightsSize = weights.size();
    file.write(reinterpret_cast<const char*>(&weightsSize), sizeof(weightsSize));
    file.write(reinterpret_cast<const char*>(weights.data()), weights.size() * sizeof(double));

    // Сохраняем словарь признаков
    size_t dictSize = featureDict.size();
    file.write(reinterpret_cast<const char*>(&dictSize), sizeof(size_t));
    for (const auto& pair : featureDict) {
        size_t keySize = pair.first.size();
        file.write(reinterpret_cast<const char*>(&keySize), sizeof(size_t));
        file.write(pair.first.c_str(), keySize);
        file.write(reinterpret_cast<const char*>(&pair.second), sizeof(size_t));
    }

    return env.Undefined();
}


Napi::Value LogisticRegression::LoadModel(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1 || !info[0].IsString()) {
        Napi::TypeError::New(env, "Expected a string").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    std::string filename = info[0].As<Napi::String>();
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        Napi::Error::New(env, "Unable to open file").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    // Загружаем веса
    size_t weightsSize;
    file.read(reinterpret_cast<char*>(&weightsSize), sizeof(weightsSize));
    weights.resize(weightsSize);
    file.read(reinterpret_cast<char*>(weights.data()), weightsSize * sizeof(double));

    // Загружаем словарь признаков
    size_t dictSize;
    file.read(reinterpret_cast<char*>(&dictSize), sizeof(size_t));
    featureDict.clear();
    for (size_t i = 0; i < dictSize; i++) {
        size_t keySize;
        file.read(reinterpret_cast<char*>(&keySize), sizeof(size_t));
        std::string key(keySize, '\0');
        file.read(&key[0], keySize);
        size_t value;
        file.read(reinterpret_cast<char*>(&value), sizeof(size_t));
        featureDict[key] = value;
    }

    return env.Undefined();
}


Napi::Object InitAll(Napi::Env env, napi_value exports) {
    Napi::Object result = Napi::Object::New(env);
    return LogisticRegression::Init(env, result);
}

NAPI_MODULE(NODE_GYP_MODULE_NAME, InitAll)