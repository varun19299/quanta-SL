n_ll = [15 31 63 127 255]';
k_ll = [11 11 10 15 13]';

% Loop through each message
message_ll = (0:2^10 - 1)';

dump_struct = struct();

for i = 1:size(n_ll)
    n = n_ll(i);
    k = k_ll(i);
    % [genpoly, t] = bchgenpoly(n, k);

    gf_message_ll = gf(de2bi(message_ll, k, 'left-msb'));
    encoded_ll = bchenc(gf_message_ll, n, k);

    code_key = sprintf("bch_%d_%d_code", n, k);
    dump_struct.(code_key) = encoded_ll == 1;
    disp(encoded_ll);

    message_key = sprintf("bch_%d_%d_message", n, k);
    dump_struct.(message_key) = gf_message_ll == 1;
end

% Save the struct
save('bch_LUT.mat', '-struct', 'dump_struct');
