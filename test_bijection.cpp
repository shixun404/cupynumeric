
#include <cstdint>
#include <iostream>

/**
 * @brief Computes the modular multiplicative inverse of n modulo 2^64.
 *
 * @param n An odd 64-bit integer.
 * @return The value x such that (n * x) mod 2^64 = 1.
 */
std::uint64_t modular_inverse_2pow64(std::uint64_t n)
{
  // Newton-Raphson iteration for finding modular inverse (mod 2^k)
  std::uint64_t x = n;
  x = x * (2 - n * x);
  x = x * (2 - n * x);
  x = x * (2 - n * x);
  x = x * (2 - n * x);
  x = x * (2 - n * x); // Five iterations are sufficient for 64 bits
  return x;
}

int main()
{
  constexpr std::uint64_t M0 = UINT64_C(0xD2E7470EE14C6C93);
  std::uint64_t M_inv = modular_inverse_2pow64(M0);

  std::cout << std::hex << "M0    = 0x" << M0 << std::endl;
  std::cout << std::hex << "M_inv = 0x" << M_inv << std::endl;
  std::cout << std::hex << "Check: M0 * M_inv = 0x" << M0 * M_inv << std::endl;

  for(uint64_t input = 0; input < 10; input++){
    std::uint64_t residual = static_cast<std::uint32_t>(M0 * input);
    std::uint64_t inverse  = static_cast<std::uint32_t>(residual * M_inv);
    std::cout << std::hex << "input: 0x" << input << std::endl;
    std::cout << std::hex << "residual: 0x" << residual << std::endl;
    std::cout << std::hex << "inverse: 0x" << inverse << std::endl;
    std::cout << std::hex << "Check: input - inverse = 0x" << input - inverse << std::endl;
  }
  return 0;
}