num_phi_samples = 128;

phi_proj = logspace(1, 8, num_phi_samples);
phi_A = logspace(0, 5, num_phi_samples);

% Number of columns in
% Conventional gray
% ie, 10-bit code here
n_col = 1024;

t_exp = 1e-4;

[phi_P_mesh, phi_A_mesh] = meshgrid(phi_proj + phi_A, phi_A);
% ij indexing
phi_P_mesh = phi_P_mesh';
phi_A_mesh = phi_A_mesh';

eval_error = zeros(size(phi_P_mesh));

% BCH_matlab encoder (15, 11, 1), (31, 11, 5), (63, 10, 13)
n = 63;
k = 10;
[genpoly, t] = bchgenpoly(n, k);

% Loop through each code
code_ll = (0:2^10 - 1)';
code_ll = de2bi(code_ll, k, 'left-msb');

% Mode [naive | avg_fixed | avg_optimal]
op_mode = "naive";

if op_mode == "naive"
    num_avg_frames = 1;
    threshold = 0.5;
elseif op_mode == "avg_fixed"
    num_avg_frames = 10;
    threshold = 0.5;
elseif op_mode == "avg_optimal"
    num_avg_frames = 10;
    threshold = optimal_threshold(phi_P_mesh, phi_A_mesh, t_exp, num_avg_frames); % 0.5;
end

for i = 1:size(code_ll, 1)
    fprintf("\nIndex %d \n", i);
    code = code_ll(i, :);
    msg = gf(code);

    % Create BCH_matlab code
    enc = bchenc(msg, n, k);

    % Replicate
    enc = reshape(enc, 1, 1, n);
    enc = repmat(enc, num_phi_samples, num_phi_samples, 1);

    % Noisy transmit
    % Phi A
    phi_A_arrived = exprnd(1 / repmat(phi_A_mesh, 1, 1, n, num_avg_frames));
    phi_A_arrived = phi_A_arrived < t_exp / num_avg_frames;
    phi_A_arrived = mean(phi_A_arrived, 4);
    phi_A_arrived = phi_A_arrived > threshold;

    % Phi P
    phi_P_arrived = exprnd(1 / repmat(phi_P_mesh, 1, 1, n, num_avg_frames));
    phi_P_arrived = phi_P_arrived < t_exp / num_avg_frames;
    phi_P_arrived = mean(phi_P_arrived, 4);
    phi_P_arrived = phi_P_arrived > threshold;

    phi_A_flips = gf(phi_A_arrived);
    phi_P_flips = gf(1 - phi_P_arrived);

    % Flip em!
    enc_0_mask = enc == 0;
    enc_1_mask = enc == 1;

    enc(enc_0_mask) = enc(enc_0_mask) + phi_A_flips(enc_0_mask);
    enc(enc_1_mask) = enc(enc_1_mask) + phi_P_flips(enc_1_mask);

    % Try decoding
    enc_flat = reshape(enc, [], n);
    [dec_flat, numerr] = bchdec(enc_flat, n, k);
    dec_flat = bi2de(dec_flat == 1, 'left-msb');
    decoded = reshape(dec_flat, num_phi_samples, num_phi_samples);

    % Find error
    binary_code = bi2de(code, 'left-msb');
    decode_error = (decoded ~= binary_code);
    decode_error_proportion = sum(decode_error, "all") / numel(decode_error);
    eval_error = eval_error + decode_error;

    fprintf("Proportion of error for code %d is %f", binary_code, decode_error_proportion);
end

eval_error = eval_error / size(code_ll, 1);

out_file = sprintf('eval_%s_bch_%d_%d_texp_%1.0e_%dx%d.mat', op_mode, n, k, t_exp, num_phi_samples, num_phi_samples);
save(out_file, 'eval_error');

function threshold = optimal_threshold(phi_P, phi_A, t_exp, num_frames)
    %myFun - Description
    %
    % Syntax: (output) = myFun(input)
    %
    % Long description
    N_p = phi_P * t_exp;
    N_a = phi_A * t_exp;

    num = N_p - N_a;

    p = 1 - exp(-N_p / num_frames);
    q = 1 - exp(-N_a / num_frames);
    denom = N_p - N_a + num_frames * (log(p) - log(q));

    threshold = num ./ denom;
end
