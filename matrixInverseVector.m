function A_inv_b = matrixInverseVector(A, b, x_init, alpha)
  % Your code here
  while true,
    x_init = x_init - alpha * 2 * A * (A*x_init -b);
    if norm(A*x_init -b)^2 < 10^-6,
      break
    endif
  endwhile
  A_inv_b = x_init
endfunction