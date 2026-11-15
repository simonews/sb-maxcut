// Exceptions of the sbmc libraries.
//
// Every error of the project derives from sbmc::Error, so that a caller can catch all of them at
// once. IoError: an input cannot be read or does not follow its format (for example a rudy file).
// InvalidArgument: a precondition of a function is violated (an invalid instance, spin vector,
// size or index); the function writes nothing to its outputs.
#pragma once

#include <stdexcept>

namespace sbmc {

class Error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

class IoError : public Error {
 public:
  using Error::Error;
};

class InvalidArgument : public Error {
 public:
  using Error::Error;
};

}  // namespace sbmc
