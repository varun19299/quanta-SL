dfile = 'gfprimdf_dump.out';
if exist(dfile, 'file'); delete(dfile); end

diary(dfile)
diary on

print_GF_primitive_poly(1, 2);
print_GF_primitive_poly(2, 2);
print_GF_primitive_poly(3, 2);
print_GF_primitive_poly(4, 2);
print_GF_primitive_poly(5, 2);
print_GF_primitive_poly(6, 2);
print_GF_primitive_poly(7, 2);
print_GF_primitive_poly(8, 2);
print_GF_primitive_poly(9, 2);
print_GF_primitive_poly(10, 2);
print_GF_primitive_poly(11, 2);
print_GF_primitive_poly(12, 2);
print_GF_primitive_poly(13, 2);
print_GF_primitive_poly(14, 2);
print_GF_primitive_poly(15, 2);
print_GF_primitive_poly(16, 2);

print_GF_primitive_poly(1, 3);
print_GF_primitive_poly(2, 3);
print_GF_primitive_poly(3, 3);
print_GF_primitive_poly(4, 3);
print_GF_primitive_poly(5, 3);
print_GF_primitive_poly(6, 3);
print_GF_primitive_poly(7, 3);
print_GF_primitive_poly(8, 3);

print_GF_primitive_poly(1, 5);
print_GF_primitive_poly(2, 5);
print_GF_primitive_poly(3, 5);
print_GF_primitive_poly(4, 5);
print_GF_primitive_poly(5, 5);
print_GF_primitive_poly(6, 5);
print_GF_primitive_poly(7, 5);
print_GF_primitive_poly(8, 5);

print_GF_primitive_poly(1, 7);
print_GF_primitive_poly(2, 7);
print_GF_primitive_poly(3, 7);
print_GF_primitive_poly(4, 7);
print_GF_primitive_poly(5, 7);
print_GF_primitive_poly(6, 7);
print_GF_primitive_poly(7, 7);
print_GF_primitive_poly(8, 7);
diary off

function primitive_poly = print_GF_primitive_poly(m, p)
    %myFun - Description
    %
    % Syntax: output = myFun(input)
    %
    % Long description
    fprintf("Primitive poly in GF(%d ^ %d). Characterisitc %d.", p, m, p)
    primitive_poly = gfprimdf(m, p);
    gfpretty(primitive_poly);
end
