patterns = [
    (r'^[A-Z].*', 'n.prop'),
    (r'^\d+$', 'num'),
    (r'^\d+\D+$', 'num'),
    (r'^\D+(l|n)ata$','n|MNT2'),
    (r'^\D+(l|n)ama$','adj|STAT'),
    (r'^\D+ra$', 'v|PFV.INTR'),
    (r'^\D+(lú|nú)$', 'prn|PL2'),
    (r'^\D+w$', 'n|PL'),
    (r'^\D+`$','n|ART'),
    (r'^\D+ba$', 'n|AUGM'),
    (r'^\D+(baa|baga)$', 'n|AG.OCC'),
    (r'^\D+bali$','ptcp|PTCP.PRIV'),
    (r'^\D+ka$', 'n|GENT'),
    (r'^\D+(la|na)$', 'n|AG.PRM'),
    (r'^\D+(lan|ran|nan)$', 'n|INSTR'),
    (r'^\D+(len|nen)$', 'ptcp|PTCP.RES'),
    (r'^\D+(li|ni)$','n|NMLZ'),
    (r'^\D+ma$','adj|COM'),
    (r'^\D+man$','adj|ADJ'),
    (r'^\D+(ma|man)\D+$', 'adj|SUPER'),
    (r'^\D+nan$', 'adj|ORD'),
    (r'^\D+nin$', 'n|DIM'),
    (r'^\D+ntan$', 'adj|PRIV'),
    (r'^\D+nci$', 'n|AG.EX'),
    (r'^\D+(ɲɔgɔn|ɲwaa?n)$', 'n|RECP'),
    (r'\D+(rɔ|nɔ)\D+$', 'adj|IN'),
    (r'^\D+ta$', 'ptcp|PTCP.POT'),
    (r'^\D+tɔ$', 'ptcp|PTCP.PROG'),
    (r'^\D+ya$', 'n|DEQU'),
    (r'^(lá|ná)\D+$', 'v|CAUS'),
    (r'^(mà|màn)\D+$', 'v|SUPER'),
    (r'(rá|rɔ́)\D+$' ,'v|IN'),
    (r'^sɔ̀\D+$' ,'v|EN'),
]
