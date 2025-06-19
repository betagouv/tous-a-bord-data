def text_to_publicode(example_tsst, example_publicode, tsst):
    return f"""
Tu es un expert en publicode.
<examples>
<example>
<example_description>
  Les différents types de tarifs sont séparés dans des fichiers dispotifs
   différents, centralisation des variables dans un fichier variables,
   et des helpers dans un fichier dédié.
</example_description>
<tsst>
{example_tsst}
</tsst>
<ideal_output>
{example_publicode}
</ideal_output>
</example>
</examples>

Inspire toi du format des fichiers fournis en exemple pour mettre en forme ces
 nouvelles informations tarifaires: {tsst} au format publicode, en
  t'inspirant des exemples fournis."
"""
