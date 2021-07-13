from simulate.scene_converter.core import pbrt_lex
import ply.yacc as yacc

# Get the token map
tokens = pbrt_lex.tokens

start = "scene"

# scene
def p_scene(t):
    """scene : directives worldblock
    | directives
    | worldblock"""

    if len(t) == 2:
        t[0] = [t[1]]
    else:
        t[0] = [t[1], t[2]]


# scene directives
def p_directives(t):
    """directives : directives directive
    | directive"""
    if len(t) > 2:
        t[0] = t[1]
        t[0].append(t[2])
    else:
        t[0] = [t[1]]

def p_directive(t):
    """directive : INTEGRATOR QUOTE SCONST QUOTE params
    | FILM QUOTE SCONST QUOTE params
    | SAMPLER QUOTE SCONST QUOTE params
    | FILTER QUOTE SCONST QUOTE params
    | CAMERA QUOTE SCONST QUOTE params
    | LOOKAT matrix
    | TRANSLATE matrix
    | ROTATE matrix
    | SCALE matrix
    | TRANSFORM matrix"""
    if len(t) == 3:
        t[0] = (t[1], None, t[2])
    else:
        t[0] = (t[1], t[3], t[5])


def p_worldblock(t):
    "worldblock : WORLDBEGIN objects WORLDEND"
    t[0] = t[2]


def p_objects(t):
    """objects : objects object
    | objects ATTRIBUTEBEGIN objects ATTRIBUTEEND
    | objects TRANSFORMBEGIN objects TRANSFORMEND
    | object"""

    if len(t) == 5:
        t[0] = t[1]
        t[0].append((t[2], t[3]))
    elif len(t) == 3:
        t[0] = t[1]
        t[0].append(t[2])
    else:
        t[0] = [t[1]]


def p_object(t):
    """object : SHAPE QUOTE SCONST QUOTE params
    | MAKENAMEDMATERIAL QUOTE SCONST QUOTE params
    | MATERIAL QUOTE SCONST QUOTE params
    | NAMEDMATERIAL QUOTE SCONST QUOTE
    | TEXTURE QUOTE SCONST QUOTE QUOTE SCONST QUOTE QUOTE SCONST QUOTE params
    | TEXTURE QUOTE SCONST QUOTE QUOTE FLOAT QUOTE QUOTE SCONST QUOTE params
    | LIGHTSOURCE QUOTE SCONST QUOTE params
    | AREALIGHTSOURCE QUOTE SCONST QUOTE params
    | LOOKAT matrix
    | TRANSLATE matrix
    | ROTATE matrix
    | SCALE matrix
    | TRANSFORM matrix
    | empty"""

    if len(t) == 2:
        t[0] = t[1]
    elif len(t) == 3:
        t[0] = (t[1], None, t[2])
    elif len(t) == 5:
        t[0] = (t[1], t[3], None)
    elif len(t) == 6:
        t[0] = (t[1], t[3], t[5])
    elif len(t) > 6:
        t[0] = (t[1], t[3], t[6], t[9], t[11])


# params and values
def p_params(t):
    """params : params param
    | param"""
    if len(t) > 2:
        t[0] = t[1]
        t[0].append(t[2])
    else:
        if t[1]:
            t[0] = [t[1]]
        else:
            t[0] = []


def p_param(t):
    """param : QUOTE INTEGER SCONST QUOTE value
    | QUOTE BOOL SCONST QUOTE value
    | QUOTE STRING SCONST QUOTE value
    | QUOTE FLOAT SCONST QUOTE value
    | QUOTE RGB SCONST QUOTE value
    | QUOTE POINT SCONST QUOTE value
    | QUOTE NORMAL SCONST QUOTE value
    | QUOTE TEX SCONST QUOTE value
    | QUOTE BLACKBODY SCONST QUOTE value
    | QUOTE SCONST SCONST QUOTE value
    | empty"""

    if len(t) > 2:
        t[0] = (t[2], t[3], t[5])


def p_value(t):
    """value : LBRACKET ICONST RBRACKET
    | LBRACKET FCONST RBRACKET
    | LBRACKET QUOTE SCONST QUOTE RBRACKET
    | LBRACKET QUOTE TRUE QUOTE RBRACKET
    | LBRACKET QUOTE FALSE QUOTE RBRACKET
    | ICONST
    | FCONST RBRACKET
    | QUOTE SCONST QUOTE
    | QUOTE TRUE QUOTE
    | QUOTE FALSE QUOTE
    | matrix
    | empty"""
    offset = int(not "[" in t)

    if len(t) > 4 - 2 * offset:
        t[0] = t[3 - offset]
    elif len(t) == 4 - 2 * offset:
        try:
            t[0] = eval(t[2 - offset])
        except (NameError, TypeError):
            t[0] = t[2 - offset]


def p_matrix(t):
    """matrix : LBRACKET numbers RBRACKET
    | numbers"""
    offset = int(not "[" in t)
    t[0] = t[2 - offset]

# Numbers
def p_numbers(t):
    """numbers : numbers number
    | number"""
    if len(t) > 2:
        t[0] = t[1]
        t[0].append(t[2])
    else:
        t[0] = [t[1]]


def p_number(t):
    """number : ICONST
    | FCONST"""

    t[0] = eval(t[1])


def p_empty(t):
    "empty :"
    pass


def p_error(t):
    print(str(t) + "Whoa. We're hosed")


# Build the grammar
parser = yacc.yacc()


def parse(data, debug=False):
    parser.error = 0
    p = parser.parse(data, debug=debug)
    if parser.error or not p:
        print(p)
        raise ValueError("Invalid pbrt file. Cannot proceed.")
    return p
