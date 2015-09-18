patterns = [
    (r'^[A-Z].*', 'n.prop'),
    (r'^\d+$', 'num'),
    (r'^\d+\D+$', 'num'),
    (r'^\D+w', 'n'), #no difference between singular and plural
    (r'^\D+na$', 'n'), #more frequent in list http://cormand.huma-num.fr/gloses.html
    (r'^\D+(l|n)ata$','n'),
    ('^\D+(l|n)ama$', 'adj'),
    (r'^\D+la$', 'n'), #more frequent in list 
    (r'^\D+ra$', 'v'),
    (r'^\D+ya$', 'n'),
    (r'^\D+bali$','ptcp'),
    (r'^\D+ka$', 'n'),
    (r'^\D+ta$', 'ptcp'),
    (r'^\D+ntan$', 'adj'),
    (r'^\D+baa$', 'n'),
    (r'^\D+baga$', 'n'),
    (r'^\D+li$','n'),
    (r'^\D+(len|nen)$', 'ptcp'),
    (r'^\D+(lan|ran|ni|ɲɔgɔn|ɲwaa?n)$', 'n'),
    (r'^\D*ɲwan', 'n'),
    (r'^\D+(man|nan)$', 'adj'),
    (r'^\D+(lan|ran)$', 'n'),
    (r'^\D+(la|na|lan|nan|ma|man)\D+$','adj'),
    (r'^\D+(l|n)en$', 'ptcp'),
    (r'\D+(rɔ|nɔ)\D+$', 'adj'),
    (r'^(la|na|ma|man|ra|rɔ|sɔ)\D+$' ,'v')
]
  
