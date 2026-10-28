"""Os templates de C usam marcadores @@TOKEN@@ em vez de f-string.

Escrever os templates como f-string significa escapar cada `{` e `}` como `{{` e `}}`,
e basta um passar batido para o arquivo sair errado sem ninguem notar. Com marcador
explicito nao ha o que escapar.
"""


def fill_template(template: str, **values) -> str:
    """Substitui @@TOKEN@@ pelo valor."""
    for key, value in values.items():
        template = template.replace(f"@@{key}@@", str(value))
    return template
