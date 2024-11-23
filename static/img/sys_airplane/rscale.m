function Nbar = rscale(A, B, C, D, K)
    % RSCALE calculates the scale factor Nbar to ensure zero steady-state error.
    % Nbar adjusts the input to match the desired output.
    % 
    % Usage:
    % Nbar = rscale(A, B, C, D, K)
    %
    % Inputs:
    %   A, B, C, D - State-space matrices
    %   K          - State feedback gain
    %
    % Output:
    %   Nbar       - Input scaling factor
    
    % Error check
    if nargin < 5
        error('Usage: Nbar = rscale(A, B, C, D, K)');
    end
    
    % Compute system size
    n = size(A, 1);
    
    % Form augmented system matrix
    A_cl = A - B * K;
    I = eye(size(A));
    
    % Compute Nbar using the formula
    Nbar = -inv(C * inv(A_cl) * B + D);
end