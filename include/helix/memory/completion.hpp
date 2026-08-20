#pragma once

namespace helix::memory {

class Completion {
public:
  virtual ~Completion() = default;

  Completion(const Completion&) = delete;
  Completion& operator=(const Completion&) = delete;
  Completion(Completion&&) = delete;
  Completion& operator=(Completion&&) = delete;

  [[nodiscard]] virtual bool ready() = 0;
  virtual void wait() = 0;

protected:
  Completion() = default;
};

} // namespace helix::memory
