
# parsetab.py
# This file is automatically generated. Do not edit.
# pylint: disable=W,C,R
_tabversion = '3.10'

_lr_method = 'LALR'

_lr_signature = 'sceneAREALIGHTSOURCE ATTRIBUTEBEGIN ATTRIBUTEEND BLACKBODY BOOL CAMERA FALSE FCONST FILM FILTER FLOAT ICONST INTEGER INTEGRATOR LBRACKET LIGHTSOURCE LOOKAT MAKENAMEDMATERIAL MATERIAL NAMEDMATERIAL NORMAL POINT QUOTE RBRACKET RGB ROTATE SAMPLER SCALE SCONST SHAPE SPECTRUM STRING TEX TEXTURE TRANSFORM TRANSFORMBEGIN TRANSFORMEND TRANSLATE TRUE WORLDBEGIN WORLDENDscene : directives worldblock\n    | directives\n    | worldblockdirectives : directives directive\n    | directivedirective : INTEGRATOR QUOTE SCONST QUOTE params\n    | FILM QUOTE SCONST QUOTE params\n    | SAMPLER QUOTE SCONST QUOTE params\n    | FILTER QUOTE SCONST QUOTE params\n    | CAMERA QUOTE SCONST QUOTE params\n    | LOOKAT matrix\n    | TRANSLATE matrix\n    | ROTATE matrix\n    | SCALE matrix\n    | TRANSFORM matrixworldblock : WORLDBEGIN objects WORLDENDobjects : objects object\n    | objects ATTRIBUTEBEGIN objects ATTRIBUTEEND\n    | objects TRANSFORMBEGIN objects TRANSFORMEND\n    | objectobject : SHAPE QUOTE SCONST QUOTE params\n    | MAKENAMEDMATERIAL QUOTE SCONST QUOTE params\n    | MATERIAL QUOTE SCONST QUOTE params\n    | NAMEDMATERIAL QUOTE SCONST QUOTE\n    | TEXTURE QUOTE SCONST QUOTE QUOTE SCONST QUOTE QUOTE SCONST QUOTE params\n    | TEXTURE QUOTE SCONST QUOTE QUOTE FLOAT QUOTE QUOTE SCONST QUOTE params\n    | LIGHTSOURCE QUOTE SCONST QUOTE params\n    | AREALIGHTSOURCE QUOTE SCONST QUOTE params\n    | LOOKAT matrix\n    | TRANSLATE matrix\n    | ROTATE matrix\n    | SCALE matrix\n    | TRANSFORM matrix\n    | emptyparams : params param\n    | paramparam : QUOTE INTEGER SCONST QUOTE value\n    | QUOTE BOOL SCONST QUOTE value\n    | QUOTE STRING SCONST QUOTE value\n    | QUOTE FLOAT SCONST QUOTE value\n    | QUOTE RGB SCONST QUOTE value\n    | QUOTE POINT SCONST QUOTE value\n    | QUOTE NORMAL SCONST QUOTE value\n    | QUOTE TEX SCONST QUOTE value\n    | QUOTE BLACKBODY SCONST QUOTE value\n    | QUOTE SCONST SCONST QUOTE value\n    | emptyvalue : LBRACKET ICONST RBRACKET\n    | LBRACKET FCONST RBRACKET\n    | LBRACKET QUOTE SCONST QUOTE RBRACKET\n    | LBRACKET QUOTE TRUE QUOTE RBRACKET\n    | LBRACKET QUOTE FALSE QUOTE RBRACKET\n    | ICONST\n    | FCONST RBRACKET\n    | QUOTE SCONST QUOTE\n    | QUOTE TRUE QUOTE\n    | QUOTE FALSE QUOTE\n    | matrix\n    | emptymatrix : LBRACKET numbers RBRACKET\n    | numbersnumbers : numbers number\n    | numbernumber : ICONST\n    | FCONSTempty :'
    
_lr_action_items = {'WORLDBEGIN':([0,2,4,17,38,40,41,42,43,44,45,46,47,70,80,81,82,83,84,85,96,97,98,99,100,101,102,119,134,135,136,137,138,139,140,141,142,143,147,149,150,151,152,153,154,155,156,157,158,159,160,161,170,173,174,175,176,177,186,187,188,],[5,5,-5,-4,-11,-61,-63,-64,-65,-12,-13,-14,-15,-62,-66,-66,-66,-66,-66,-60,-6,-36,-47,-7,-8,-9,-10,-35,-66,-66,-66,-66,-66,-66,-66,-66,-66,-66,-37,-53,-65,-58,-59,-46,-38,-39,-40,-41,-42,-43,-44,-45,-54,-55,-56,-57,-48,-49,-50,-51,-52,]),'INTEGRATOR':([0,2,4,17,38,40,41,42,43,44,45,46,47,70,80,81,82,83,84,85,96,97,98,99,100,101,102,119,134,135,136,137,138,139,140,141,142,143,147,149,150,151,152,153,154,155,156,157,158,159,160,161,170,173,174,175,176,177,186,187,188,],[6,6,-5,-4,-11,-61,-63,-64,-65,-12,-13,-14,-15,-62,-66,-66,-66,-66,-66,-60,-6,-36,-47,-7,-8,-9,-10,-35,-66,-66,-66,-66,-66,-66,-66,-66,-66,-66,-37,-53,-65,-58,-59,-46,-38,-39,-40,-41,-42,-43,-44,-45,-54,-55,-56,-57,-48,-49,-50,-51,-52,]),'FILM':([0,2,4,17,38,40,41,42,43,44,45,46,47,70,80,81,82,83,84,85,96,97,98,99,100,101,102,119,134,135,136,137,138,139,140,141,142,143,147,149,150,151,152,153,154,155,156,157,158,159,160,161,170,173,174,175,176,177,186,187,188,],[7,7,-5,-4,-11,-61,-63,-64,-65,-12,-13,-14,-15,-62,-66,-66,-66,-66,-66,-60,-6,-36,-47,-7,-8,-9,-10,-35,-66,-66,-66,-66,-66,-66,-66,-66,-66,-66,-37,-53,-65,-58,-59,-46,-38,-39,-40,-41,-42,-43,-44,-45,-54,-55,-56,-57,-48,-49,-50,-51,-52,]),'SAMPLER':([0,2,4,17,38,40,41,42,43,44,45,46,47,70,80,81,82,83,84,85,96,97,98,99,100,101,102,119,134,135,136,137,138,139,140,141,142,143,147,149,150,151,152,153,154,155,156,157,158,159,160,161,170,173,174,175,176,177,186,187,188,],[8,8,-5,-4,-11,-61,-63,-64,-65,-12,-13,-14,-15,-62,-66,-66,-66,-66,-66,-60,-6,-36,-47,-7,-8,-9,-10,-35,-66,-66,-66,-66,-66,-66,-66,-66,-66,-66,-37,-53,-65,-58,-59,-46,-38,-39,-40,-41,-42,-43,-44,-45,-54,-55,-56,-57,-48,-49,-50,-51,-52,]),'FILTER':([0,2,4,17,38,40,41,42,43,44,45,46,47,70,80,81,82,83,84,85,96,97,98,99,100,101,102,119,134,135,136,137,138,139,140,141,142,143,147,149,150,151,152,153,154,155,156,157,158,159,160,161,170,173,174,175,176,177,186,187,188,],[9,9,-5,-4,-11,-61,-63,-64,-65,-12,-13,-14,-15,-62,-66,-66,-66,-66,-66,-60,-6,-36,-47,-7,-8,-9,-10,-35,-66,-66,-66,-66,-66,-66,-66,-66,-66,-66,-37,-53,-65,-58,-59,-46,-38,-39,-40,-41,-42,-43,-44,-45,-54,-55,-56,-57,-48,-49,-50,-51,-52,]),'CAMERA':([0,2,4,17,38,40,41,42,43,44,45,46,47,70,80,81,82,83,84,85,96,97,98,99,100,101,102,119,134,135,136,137,138,139,140,141,142,143,147,149,150,151,152,153,154,155,156,157,158,159,160,161,170,173,174,175,176,177,186,187,188,],[10,10,-5,-4,-11,-61,-63,-64,-65,-12,-13,-14,-15,-62,-66,-66,-66,-66,-66,-60,-6,-36,-47,-7,-8,-9,-10,-35,-66,-66,-66,-66,-66,-66,-66,-66,-66,-66,-37,-53,-65,-58,-59,-46,-38,-39,-40,-41,-42,-43,-44,-45,-54,-55,-56,-57,-48,-49,-50,-51,-52,]),'LOOKAT':([0,2,4,5,17,18,19,32,38,40,41,42,43,44,45,46,47,49,50,51,59,60,61,62,63,70,71,72,80,81,82,83,84,85,86,87,88,89,90,91,93,94,96,97,98,99,100,101,102,103,104,105,107,108,119,134,135,136,137,138,139,140,141,142,143,147,149,150,151,152,153,154,155,156,157,158,159,160,161,170,171,172,173,174,175,176,177,181,182,186,187,188,],[11,11,-5,27,-4,27,-20,-34,-11,-61,-63,-64,-65,-12,-13,-14,-15,-17,27,27,-29,-30,-31,-32,-33,-62,27,27,-66,-66,-66,-66,-66,-60,-18,-19,-66,-66,-66,-24,-66,-66,-6,-36,-47,-7,-8,-9,-10,-21,-22,-23,-27,-28,-35,-66,-66,-66,-66,-66,-66,-66,-66,-66,-66,-37,-53,-65,-58,-59,-46,-38,-39,-40,-41,-42,-43,-44,-45,-54,-66,-66,-55,-56,-57,-48,-49,-25,-26,-50,-51,-52,]),'TRANSLATE':([0,2,4,5,17,18,19,32,38,40,41,42,43,44,45,46,47,49,50,51,59,60,61,62,63,70,71,72,80,81,82,83,84,85,86,87,88,89,90,91,93,94,96,97,98,99,100,101,102,103,104,105,107,108,119,134,135,136,137,138,139,140,141,142,143,147,149,150,151,152,153,154,155,156,157,158,159,160,161,170,171,172,173,174,175,176,177,181,182,186,187,188,],[12,12,-5,28,-4,28,-20,-34,-11,-61,-63,-64,-65,-12,-13,-14,-15,-17,28,28,-29,-30,-31,-32,-33,-62,28,28,-66,-66,-66,-66,-66,-60,-18,-19,-66,-66,-66,-24,-66,-66,-6,-36,-47,-7,-8,-9,-10,-21,-22,-23,-27,-28,-35,-66,-66,-66,-66,-66,-66,-66,-66,-66,-66,-37,-53,-65,-58,-59,-46,-38,-39,-40,-41,-42,-43,-44,-45,-54,-66,-66,-55,-56,-57,-48,-49,-25,-26,-50,-51,-52,]),'ROTATE':([0,2,4,5,17,18,19,32,38,40,41,42,43,44,45,46,47,49,50,51,59,60,61,62,63,70,71,72,80,81,82,83,84,85,86,87,88,89,90,91,93,94,96,97,98,99,100,101,102,103,104,105,107,108,119,134,135,136,137,138,139,140,141,142,143,147,149,150,151,152,153,154,155,156,157,158,159,160,161,170,171,172,173,174,175,176,177,181,182,186,187,188,],[13,13,-5,29,-4,29,-20,-34,-11,-61,-63,-64,-65,-12,-13,-14,-15,-17,29,29,-29,-30,-31,-32,-33,-62,29,29,-66,-66,-66,-66,-66,-60,-18,-19,-66,-66,-66,-24,-66,-66,-6,-36,-47,-7,-8,-9,-10,-21,-22,-23,-27,-28,-35,-66,-66,-66,-66,-66,-66,-66,-66,-66,-66,-37,-53,-65,-58,-59,-46,-38,-39,-40,-41,-42,-43,-44,-45,-54,-66,-66,-55,-56,-57,-48,-49,-25,-26,-50,-51,-52,]),'SCALE':([0,2,4,5,17,18,19,32,38,40,41,42,43,44,45,46,47,49,50,51,59,60,61,62,63,70,71,72,80,81,82,83,84,85,86,87,88,89,90,91,93,94,96,97,98,99,100,101,102,103,104,105,107,108,119,134,135,136,137,138,139,140,141,142,143,147,149,150,151,152,153,154,155,156,157,158,159,160,161,170,171,172,173,174,175,176,177,181,182,186,187,188,],[14,14,-5,30,-4,30,-20,-34,-11,-61,-63,-64,-65,-12,-13,-14,-15,-17,30,30,-29,-30,-31,-32,-33,-62,30,30,-66,-66,-66,-66,-66,-60,-18,-19,-66,-66,-66,-24,-66,-66,-6,-36,-47,-7,-8,-9,-10,-21,-22,-23,-27,-28,-35,-66,-66,-66,-66,-66,-66,-66,-66,-66,-66,-37,-53,-65,-58,-59,-46,-38,-39,-40,-41,-42,-43,-44,-45,-54,-66,-66,-55,-56,-57,-48,-49,-25,-26,-50,-51,-52,]),'TRANSFORM':([0,2,4,5,17,18,19,32,38,40,41,42,43,44,45,46,47,49,50,51,59,60,61,62,63,70,71,72,80,81,82,83,84,85,86,87,88,89,90,91,93,94,96,97,98,99,100,101,102,103,104,105,107,108,119,134,135,136,137,138,139,140,141,142,143,147,149,150,151,152,153,154,155,156,157,158,159,160,161,170,171,172,173,174,175,176,177,181,182,186,187,188,],[15,15,-5,31,-4,31,-20,-34,-11,-61,-63,-64,-65,-12,-13,-14,-15,-17,31,31,-29,-30,-31,-32,-33,-62,31,31,-66,-66,-66,-66,-66,-60,-18,-19,-66,-66,-66,-24,-66,-66,-6,-36,-47,-7,-8,-9,-10,-21,-22,-23,-27,-28,-35,-66,-66,-66,-66,-66,-66,-66,-66,-66,-66,-37,-53,-65,-58,-59,-46,-38,-39,-40,-41,-42,-43,-44,-45,-54,-66,-66,-55,-56,-57,-48,-49,-25,-26,-50,-51,-52,]),'$end':([1,2,3,4,16,17,38,40,41,42,43,44,45,46,47,48,70,80,81,82,83,84,85,96,97,98,99,100,101,102,119,134,135,136,137,138,139,140,141,142,143,147,149,150,151,152,153,154,155,156,157,158,159,160,161,170,173,174,175,176,177,186,187,188,],[0,-2,-3,-5,-1,-4,-11,-61,-63,-64,-65,-12,-13,-14,-15,-16,-62,-66,-66,-66,-66,-66,-60,-6,-36,-47,-7,-8,-9,-10,-35,-66,-66,-66,-66,-66,-66,-66,-66,-66,-66,-37,-53,-65,-58,-59,-46,-38,-39,-40,-41,-42,-43,-44,-45,-54,-55,-56,-57,-48,-49,-50,-51,-52,]),'SHAPE':([5,18,19,32,40,41,42,43,49,50,51,59,60,61,62,63,70,71,72,85,86,87,88,89,90,91,93,94,97,98,103,104,105,107,108,119,134,135,136,137,138,139,140,141,142,143,147,149,150,151,152,153,154,155,156,157,158,159,160,161,170,171,172,173,174,175,176,177,181,182,186,187,188,],[20,20,-20,-34,-61,-63,-64,-65,-17,20,20,-29,-30,-31,-32,-33,-62,20,20,-60,-18,-19,-66,-66,-66,-24,-66,-66,-36,-47,-21,-22,-23,-27,-28,-35,-66,-66,-66,-66,-66,-66,-66,-66,-66,-66,-37,-53,-65,-58,-59,-46,-38,-39,-40,-41,-42,-43,-44,-45,-54,-66,-66,-55,-56,-57,-48,-49,-25,-26,-50,-51,-52,]),'MAKENAMEDMATERIAL':([5,18,19,32,40,41,42,43,49,50,51,59,60,61,62,63,70,71,72,85,86,87,88,89,90,91,93,94,97,98,103,104,105,107,108,119,134,135,136,137,138,139,140,141,142,143,147,149,150,151,152,153,154,155,156,157,158,159,160,161,170,171,172,173,174,175,176,177,181,182,186,187,188,],[21,21,-20,-34,-61,-63,-64,-65,-17,21,21,-29,-30,-31,-32,-33,-62,21,21,-60,-18,-19,-66,-66,-66,-24,-66,-66,-36,-47,-21,-22,-23,-27,-28,-35,-66,-66,-66,-66,-66,-66,-66,-66,-66,-66,-37,-53,-65,-58,-59,-46,-38,-39,-40,-41,-42,-43,-44,-45,-54,-66,-66,-55,-56,-57,-48,-49,-25,-26,-50,-51,-52,]),'MATERIAL':([5,18,19,32,40,41,42,43,49,50,51,59,60,61,62,63,70,71,72,85,86,87,88,89,90,91,93,94,97,98,103,104,105,107,108,119,134,135,136,137,138,139,140,141,142,143,147,149,150,151,152,153,154,155,156,157,158,159,160,161,170,171,172,173,174,175,176,177,181,182,186,187,188,],[22,22,-20,-34,-61,-63,-64,-65,-17,22,22,-29,-30,-31,-32,-33,-62,22,22,-60,-18,-19,-66,-66,-66,-24,-66,-66,-36,-47,-21,-22,-23,-27,-28,-35,-66,-66,-66,-66,-66,-66,-66,-66,-66,-66,-37,-53,-65,-58,-59,-46,-38,-39,-40,-41,-42,-43,-44,-45,-54,-66,-66,-55,-56,-57,-48,-49,-25,-26,-50,-51,-52,]),'NAMEDMATERIAL':([5,18,19,32,40,41,42,43,49,50,51,59,60,61,62,63,70,71,72,85,86,87,88,89,90,91,93,94,97,98,103,104,105,107,108,119,134,135,136,137,138,139,140,141,142,143,147,149,150,151,152,153,154,155,156,157,158,159,160,161,170,171,172,173,174,175,176,177,181,182,186,187,188,],[23,23,-20,-34,-61,-63,-64,-65,-17,23,23,-29,-30,-31,-32,-33,-62,23,23,-60,-18,-19,-66,-66,-66,-24,-66,-66,-36,-47,-21,-22,-23,-27,-28,-35,-66,-66,-66,-66,-66,-66,-66,-66,-66,-66,-37,-53,-65,-58,-59,-46,-38,-39,-40,-41,-42,-43,-44,-45,-54,-66,-66,-55,-56,-57,-48,-49,-25,-26,-50,-51,-52,]),'TEXTURE':([5,18,19,32,40,41,42,43,49,50,51,59,60,61,62,63,70,71,72,85,86,87,88,89,90,91,93,94,97,98,103,104,105,107,108,119,134,135,136,137,138,139,140,141,142,143,147,149,150,151,152,153,154,155,156,157,158,159,160,161,170,171,172,173,174,175,176,177,181,182,186,187,188,],[24,24,-20,-34,-61,-63,-64,-65,-17,24,24,-29,-30,-31,-32,-33,-62,24,24,-60,-18,-19,-66,-66,-66,-24,-66,-66,-36,-47,-21,-22,-23,-27,-28,-35,-66,-66,-66,-66,-66,-66,-66,-66,-66,-66,-37,-53,-65,-58,-59,-46,-38,-39,-40,-41,-42,-43,-44,-45,-54,-66,-66,-55,-56,-57,-48,-49,-25,-26,-50,-51,-52,]),'LIGHTSOURCE':([5,18,19,32,40,41,42,43,49,50,51,59,60,61,62,63,70,71,72,85,86,87,88,89,90,91,93,94,97,98,103,104,105,107,108,119,134,135,136,137,138,139,140,141,142,143,147,149,150,151,152,153,154,155,156,157,158,159,160,161,170,171,172,173,174,175,176,177,181,182,186,187,188,],[25,25,-20,-34,-61,-63,-64,-65,-17,25,25,-29,-30,-31,-32,-33,-62,25,25,-60,-18,-19,-66,-66,-66,-24,-66,-66,-36,-47,-21,-22,-23,-27,-28,-35,-66,-66,-66,-66,-66,-66,-66,-66,-66,-66,-37,-53,-65,-58,-59,-46,-38,-39,-40,-41,-42,-43,-44,-45,-54,-66,-66,-55,-56,-57,-48,-49,-25,-26,-50,-51,-52,]),'AREALIGHTSOURCE':([5,18,19,32,40,41,42,43,49,50,51,59,60,61,62,63,70,71,72,85,86,87,88,89,90,91,93,94,97,98,103,104,105,107,108,119,134,135,136,137,138,139,140,141,142,143,147,149,150,151,152,153,154,155,156,157,158,159,160,161,170,171,172,173,174,175,176,177,181,182,186,187,188,],[26,26,-20,-34,-61,-63,-64,-65,-17,26,26,-29,-30,-31,-32,-33,-62,26,26,-60,-18,-19,-66,-66,-66,-24,-66,-66,-36,-47,-21,-22,-23,-27,-28,-35,-66,-66,-66,-66,-66,-66,-66,-66,-66,-66,-37,-53,-65,-58,-59,-46,-38,-39,-40,-41,-42,-43,-44,-45,-54,-66,-66,-55,-56,-57,-48,-49,-25,-26,-50,-51,-52,]),'WORLDEND':([5,18,19,32,40,41,42,43,49,59,60,61,62,63,70,85,86,87,88,89,90,91,93,94,97,98,103,104,105,107,108,119,134,135,136,137,138,139,140,141,142,143,147,149,150,151,152,153,154,155,156,157,158,159,160,161,170,171,172,173,174,175,176,177,181,182,186,187,188,],[-66,48,-20,-34,-61,-63,-64,-65,-17,-29,-30,-31,-32,-33,-62,-60,-18,-19,-66,-66,-66,-24,-66,-66,-36,-47,-21,-22,-23,-27,-28,-35,-66,-66,-66,-66,-66,-66,-66,-66,-66,-66,-37,-53,-65,-58,-59,-46,-38,-39,-40,-41,-42,-43,-44,-45,-54,-66,-66,-55,-56,-57,-48,-49,-25,-26,-50,-51,-52,]),'ATTRIBUTEBEGIN':([5,18,19,32,40,41,42,43,49,50,51,59,60,61,62,63,70,71,72,85,86,87,88,89,90,91,93,94,97,98,103,104,105,107,108,119,134,135,136,137,138,139,140,141,142,143,147,149,150,151,152,153,154,155,156,157,158,159,160,161,170,171,172,173,174,175,176,177,181,182,186,187,188,],[-66,50,-20,-34,-61,-63,-64,-65,-17,-66,-66,-29,-30,-31,-32,-33,-62,50,50,-60,-18,-19,-66,-66,-66,-24,-66,-66,-36,-47,-21,-22,-23,-27,-28,-35,-66,-66,-66,-66,-66,-66,-66,-66,-66,-66,-37,-53,-65,-58,-59,-46,-38,-39,-40,-41,-42,-43,-44,-45,-54,-66,-66,-55,-56,-57,-48,-49,-25,-26,-50,-51,-52,]),'TRANSFORMBEGIN':([5,18,19,32,40,41,42,43,49,50,51,59,60,61,62,63,70,71,72,85,86,87,88,89,90,91,93,94,97,98,103,104,105,107,108,119,134,135,136,137,138,139,140,141,142,143,147,149,150,151,152,153,154,155,156,157,158,159,160,161,170,171,172,173,174,175,176,177,181,182,186,187,188,],[-66,51,-20,-34,-61,-63,-64,-65,-17,-66,-66,-29,-30,-31,-32,-33,-62,51,51,-60,-18,-19,-66,-66,-66,-24,-66,-66,-36,-47,-21,-22,-23,-27,-28,-35,-66,-66,-66,-66,-66,-66,-66,-66,-66,-66,-37,-53,-65,-58,-59,-46,-38,-39,-40,-41,-42,-43,-44,-45,-54,-66,-66,-55,-56,-57,-48,-49,-25,-26,-50,-51,-52,]),'QUOTE':([6,7,8,9,10,20,21,22,23,24,25,26,40,41,42,43,64,65,66,67,68,70,73,74,75,76,77,78,79,80,81,82,83,84,85,88,89,90,92,93,94,96,97,98,99,100,101,102,103,104,105,107,108,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,170,171,172,173,174,175,176,177,178,179,180,181,182,186,187,188,],[33,34,35,36,37,52,53,54,55,56,57,58,-61,-63,-64,-65,80,81,82,83,84,-62,88,89,90,91,92,93,94,95,95,95,95,95,-60,95,95,95,106,95,95,95,-36,-47,95,95,95,95,95,95,95,95,95,-35,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,146,146,146,146,146,146,146,146,146,-37,169,-53,-65,-58,-59,-46,-38,-39,-40,-41,-42,-43,-44,-45,171,172,173,174,175,-54,95,95,-55,-56,-57,-48,-49,183,184,185,95,95,-50,-51,-52,]),'LBRACKET':([11,12,13,14,15,27,28,29,30,31,134,135,136,137,138,139,140,141,142,143,],[39,39,39,39,39,39,39,39,39,39,148,148,148,148,148,148,148,148,148,148,]),'ICONST':([11,12,13,14,15,27,28,29,30,31,39,40,41,42,43,69,70,134,135,136,137,138,139,140,141,142,143,148,149,150,167,168,],[42,42,42,42,42,42,42,42,42,42,42,42,-63,-64,-65,42,-62,149,149,149,149,149,149,149,149,149,149,167,-64,-65,-64,-65,]),'FCONST':([11,12,13,14,15,27,28,29,30,31,39,40,41,42,43,69,70,134,135,136,137,138,139,140,141,142,143,148,149,150,167,168,],[43,43,43,43,43,43,43,43,43,43,43,43,-63,-64,-65,43,-62,150,150,150,150,150,150,150,150,150,150,168,-64,-65,-64,-65,]),'ATTRIBUTEEND':([19,32,40,41,42,43,49,50,59,60,61,62,63,70,71,85,86,87,88,89,90,91,93,94,97,98,103,104,105,107,108,119,134,135,136,137,138,139,140,141,142,143,147,149,150,151,152,153,154,155,156,157,158,159,160,161,170,171,172,173,174,175,176,177,181,182,186,187,188,],[-20,-34,-61,-63,-64,-65,-17,-66,-29,-30,-31,-32,-33,-62,86,-60,-18,-19,-66,-66,-66,-24,-66,-66,-36,-47,-21,-22,-23,-27,-28,-35,-66,-66,-66,-66,-66,-66,-66,-66,-66,-66,-37,-53,-65,-58,-59,-46,-38,-39,-40,-41,-42,-43,-44,-45,-54,-66,-66,-55,-56,-57,-48,-49,-25,-26,-50,-51,-52,]),'TRANSFORMEND':([19,32,40,41,42,43,49,51,59,60,61,62,63,70,72,85,86,87,88,89,90,91,93,94,97,98,103,104,105,107,108,119,134,135,136,137,138,139,140,141,142,143,147,149,150,151,152,153,154,155,156,157,158,159,160,161,170,171,172,173,174,175,176,177,181,182,186,187,188,],[-20,-34,-61,-63,-64,-65,-17,-66,-29,-30,-31,-32,-33,-62,87,-60,-18,-19,-66,-66,-66,-24,-66,-66,-36,-47,-21,-22,-23,-27,-28,-35,-66,-66,-66,-66,-66,-66,-66,-66,-66,-66,-37,-53,-65,-58,-59,-46,-38,-39,-40,-41,-42,-43,-44,-45,-54,-66,-66,-55,-56,-57,-48,-49,-25,-26,-50,-51,-52,]),'SCONST':([33,34,35,36,37,52,53,54,55,56,57,58,95,106,109,110,111,112,113,114,115,116,117,118,144,145,146,169,],[64,65,66,67,68,73,74,75,76,77,78,79,110,120,122,123,124,125,126,127,128,129,130,131,162,163,164,178,]),'RBRACKET':([41,42,43,69,70,150,167,168,183,184,185,],[-63,-64,-65,85,-62,170,176,177,186,187,188,]),'INTEGER':([95,],[109,]),'BOOL':([95,],[111,]),'STRING':([95,],[112,]),'FLOAT':([95,106,],[113,121,]),'RGB':([95,],[114,]),'POINT':([95,],[115,]),'NORMAL':([95,],[116,]),'TEX':([95,],[117,]),'BLACKBODY':([95,],[118,]),'TRUE':([146,169,],[165,179,]),'FALSE':([146,169,],[166,180,]),}

_lr_action = {}
for _k, _v in _lr_action_items.items():
   for _x,_y in zip(_v[0],_v[1]):
      if not _x in _lr_action:  _lr_action[_x] = {}
      _lr_action[_x][_k] = _y
del _lr_action_items

_lr_goto_items = {'scene':([0,],[1,]),'directives':([0,],[2,]),'worldblock':([0,2,],[3,16,]),'directive':([0,2,],[4,17,]),'objects':([5,50,51,],[18,71,72,]),'object':([5,18,50,51,71,72,],[19,49,19,19,49,49,]),'empty':([5,18,50,51,71,72,80,81,82,83,84,88,89,90,93,94,96,99,100,101,102,103,104,105,107,108,134,135,136,137,138,139,140,141,142,143,171,172,181,182,],[32,32,32,32,32,32,98,98,98,98,98,98,98,98,98,98,98,98,98,98,98,98,98,98,98,98,152,152,152,152,152,152,152,152,152,152,98,98,98,98,]),'matrix':([11,12,13,14,15,27,28,29,30,31,134,135,136,137,138,139,140,141,142,143,],[38,44,45,46,47,59,60,61,62,63,151,151,151,151,151,151,151,151,151,151,]),'numbers':([11,12,13,14,15,27,28,29,30,31,39,134,135,136,137,138,139,140,141,142,143,148,],[40,40,40,40,40,40,40,40,40,40,69,40,40,40,40,40,40,40,40,40,40,69,]),'number':([11,12,13,14,15,27,28,29,30,31,39,40,69,134,135,136,137,138,139,140,141,142,143,148,],[41,41,41,41,41,41,41,41,41,41,41,70,70,41,41,41,41,41,41,41,41,41,41,41,]),'params':([80,81,82,83,84,88,89,90,93,94,171,172,],[96,99,100,101,102,103,104,105,107,108,181,182,]),'param':([80,81,82,83,84,88,89,90,93,94,96,99,100,101,102,103,104,105,107,108,171,172,181,182,],[97,97,97,97,97,97,97,97,97,97,119,119,119,119,119,119,119,119,119,119,97,97,119,119,]),'value':([134,135,136,137,138,139,140,141,142,143,],[147,153,154,155,156,157,158,159,160,161,]),}

_lr_goto = {}
for _k, _v in _lr_goto_items.items():
   for _x, _y in zip(_v[0], _v[1]):
       if not _x in _lr_goto: _lr_goto[_x] = {}
       _lr_goto[_x][_k] = _y
del _lr_goto_items
_lr_productions = [
  ("S' -> scene","S'",1,None,None,None),
  ('scene -> directives worldblock','scene',2,'p_scene','pbrt_yacc.py',11),
  ('scene -> directives','scene',1,'p_scene','pbrt_yacc.py',12),
  ('scene -> worldblock','scene',1,'p_scene','pbrt_yacc.py',13),
  ('directives -> directives directive','directives',2,'p_directives','pbrt_yacc.py',23),
  ('directives -> directive','directives',1,'p_directives','pbrt_yacc.py',24),
  ('directive -> INTEGRATOR QUOTE SCONST QUOTE params','directive',5,'p_directive','pbrt_yacc.py',33),
  ('directive -> FILM QUOTE SCONST QUOTE params','directive',5,'p_directive','pbrt_yacc.py',34),
  ('directive -> SAMPLER QUOTE SCONST QUOTE params','directive',5,'p_directive','pbrt_yacc.py',35),
  ('directive -> FILTER QUOTE SCONST QUOTE params','directive',5,'p_directive','pbrt_yacc.py',36),
  ('directive -> CAMERA QUOTE SCONST QUOTE params','directive',5,'p_directive','pbrt_yacc.py',37),
  ('directive -> LOOKAT matrix','directive',2,'p_directive','pbrt_yacc.py',38),
  ('directive -> TRANSLATE matrix','directive',2,'p_directive','pbrt_yacc.py',39),
  ('directive -> ROTATE matrix','directive',2,'p_directive','pbrt_yacc.py',40),
  ('directive -> SCALE matrix','directive',2,'p_directive','pbrt_yacc.py',41),
  ('directive -> TRANSFORM matrix','directive',2,'p_directive','pbrt_yacc.py',42),
  ('worldblock -> WORLDBEGIN objects WORLDEND','worldblock',3,'p_worldblock','pbrt_yacc.py',50),
  ('objects -> objects object','objects',2,'p_objects','pbrt_yacc.py',55),
  ('objects -> objects ATTRIBUTEBEGIN objects ATTRIBUTEEND','objects',4,'p_objects','pbrt_yacc.py',56),
  ('objects -> objects TRANSFORMBEGIN objects TRANSFORMEND','objects',4,'p_objects','pbrt_yacc.py',57),
  ('objects -> object','objects',1,'p_objects','pbrt_yacc.py',58),
  ('object -> SHAPE QUOTE SCONST QUOTE params','object',5,'p_object','pbrt_yacc.py',71),
  ('object -> MAKENAMEDMATERIAL QUOTE SCONST QUOTE params','object',5,'p_object','pbrt_yacc.py',72),
  ('object -> MATERIAL QUOTE SCONST QUOTE params','object',5,'p_object','pbrt_yacc.py',73),
  ('object -> NAMEDMATERIAL QUOTE SCONST QUOTE','object',4,'p_object','pbrt_yacc.py',74),
  ('object -> TEXTURE QUOTE SCONST QUOTE QUOTE SCONST QUOTE QUOTE SCONST QUOTE params','object',11,'p_object','pbrt_yacc.py',75),
  ('object -> TEXTURE QUOTE SCONST QUOTE QUOTE FLOAT QUOTE QUOTE SCONST QUOTE params','object',11,'p_object','pbrt_yacc.py',76),
  ('object -> LIGHTSOURCE QUOTE SCONST QUOTE params','object',5,'p_object','pbrt_yacc.py',77),
  ('object -> AREALIGHTSOURCE QUOTE SCONST QUOTE params','object',5,'p_object','pbrt_yacc.py',78),
  ('object -> LOOKAT matrix','object',2,'p_object','pbrt_yacc.py',79),
  ('object -> TRANSLATE matrix','object',2,'p_object','pbrt_yacc.py',80),
  ('object -> ROTATE matrix','object',2,'p_object','pbrt_yacc.py',81),
  ('object -> SCALE matrix','object',2,'p_object','pbrt_yacc.py',82),
  ('object -> TRANSFORM matrix','object',2,'p_object','pbrt_yacc.py',83),
  ('object -> empty','object',1,'p_object','pbrt_yacc.py',84),
  ('params -> params param','params',2,'p_params','pbrt_yacc.py',100),
  ('params -> param','params',1,'p_params','pbrt_yacc.py',101),
  ('param -> QUOTE INTEGER SCONST QUOTE value','param',5,'p_param','pbrt_yacc.py',113),
  ('param -> QUOTE BOOL SCONST QUOTE value','param',5,'p_param','pbrt_yacc.py',114),
  ('param -> QUOTE STRING SCONST QUOTE value','param',5,'p_param','pbrt_yacc.py',115),
  ('param -> QUOTE FLOAT SCONST QUOTE value','param',5,'p_param','pbrt_yacc.py',116),
  ('param -> QUOTE RGB SCONST QUOTE value','param',5,'p_param','pbrt_yacc.py',117),
  ('param -> QUOTE POINT SCONST QUOTE value','param',5,'p_param','pbrt_yacc.py',118),
  ('param -> QUOTE NORMAL SCONST QUOTE value','param',5,'p_param','pbrt_yacc.py',119),
  ('param -> QUOTE TEX SCONST QUOTE value','param',5,'p_param','pbrt_yacc.py',120),
  ('param -> QUOTE BLACKBODY SCONST QUOTE value','param',5,'p_param','pbrt_yacc.py',121),
  ('param -> QUOTE SCONST SCONST QUOTE value','param',5,'p_param','pbrt_yacc.py',122),
  ('param -> empty','param',1,'p_param','pbrt_yacc.py',123),
  ('value -> LBRACKET ICONST RBRACKET','value',3,'p_value','pbrt_yacc.py',130),
  ('value -> LBRACKET FCONST RBRACKET','value',3,'p_value','pbrt_yacc.py',131),
  ('value -> LBRACKET QUOTE SCONST QUOTE RBRACKET','value',5,'p_value','pbrt_yacc.py',132),
  ('value -> LBRACKET QUOTE TRUE QUOTE RBRACKET','value',5,'p_value','pbrt_yacc.py',133),
  ('value -> LBRACKET QUOTE FALSE QUOTE RBRACKET','value',5,'p_value','pbrt_yacc.py',134),
  ('value -> ICONST','value',1,'p_value','pbrt_yacc.py',135),
  ('value -> FCONST RBRACKET','value',2,'p_value','pbrt_yacc.py',136),
  ('value -> QUOTE SCONST QUOTE','value',3,'p_value','pbrt_yacc.py',137),
  ('value -> QUOTE TRUE QUOTE','value',3,'p_value','pbrt_yacc.py',138),
  ('value -> QUOTE FALSE QUOTE','value',3,'p_value','pbrt_yacc.py',139),
  ('value -> matrix','value',1,'p_value','pbrt_yacc.py',140),
  ('value -> empty','value',1,'p_value','pbrt_yacc.py',141),
  ('matrix -> LBRACKET numbers RBRACKET','matrix',3,'p_matrix','pbrt_yacc.py',154),
  ('matrix -> numbers','matrix',1,'p_matrix','pbrt_yacc.py',155),
  ('numbers -> numbers number','numbers',2,'p_numbers','pbrt_yacc.py',161),
  ('numbers -> number','numbers',1,'p_numbers','pbrt_yacc.py',162),
  ('number -> ICONST','number',1,'p_number','pbrt_yacc.py',171),
  ('number -> FCONST','number',1,'p_number','pbrt_yacc.py',172),
  ('empty -> <empty>','empty',0,'p_empty','pbrt_yacc.py',178),
]
