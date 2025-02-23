{
  "targets": [
    {
      "target_name": "logistic_regression_classifier",
      "sources": [ "src/logistic_regression_classifier.cpp" ],
      "include_dirs": [
        "node_modules/node-addon-api"
      ],
      "dependencies": [
        "<!(node -p \"require('node-addon-api').gyp\")"
      ],
      "cflags!": [ "-fno-exceptions" ],
      "cflags_cc!": [ "-fno-exceptions" ],
      "xcode_settings": {
        "OTHER_CFLAGS": [ "-fexceptions" ],
        "OTHER_CPLUSPLUSFLAGS": [ "-fexceptions" ]
      },
      "msvs_settings": {
        "VCCLCompilerTool": { "ExceptionHandling": 1 }
      },
      "defines": [ "NAPI_DISABLE_CPP_EXCEPTIONS" ]
    }
  ]
}
