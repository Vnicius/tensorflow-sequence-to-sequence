# arquivo = 'europarl-v7.pt-en.pt'
arquivo = 'europarl-v7.pt-en.en'

with open(arquivo, 'r') as f:
    out = open(arquivo + '.trunc', 'w')

    for line in f:
        sp = line.split(' ');
        if len(sp) == 20:
            out.write(' '.join(sp))
        elif len(sp) > 20:
            out.write(' '.join(sp[:20]))