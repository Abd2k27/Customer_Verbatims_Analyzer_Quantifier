"""
Générateur de dataset multilingue de test.

Prend 100 avis réels en anglais du fichier bank_reviews_sample.csv
et génère 400 avis supplémentaires en français, espagnol, portugais, allemand
et belge (français/néerlandais avec belgicismes comme Bancontact, gsm, septante)
traitant de sujets bancaires divers avec des sources variées.

Le fichier final contient exactement 500 lignes.
"""

import sys
import random
from pathlib import Path

# Ajouter le root du projet au path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from config.settings import RAW_DIR

# Définition des sources possibles
SOURCES = ["trustpilot", "app_store", "enquete_satisfaction", "google_reviews", "support_ticket"]

# --- TEMPLATES DE VERBATIMS PAR LANGUE ---

# 1. Français (fr) - 100 verbatims cibles
FR_TEMPLATES = [
    # Négatifs
    "L'application mobile crash à chaque fois que je tente d'effectuer un virement instantané. C'est inadmissible pour une grande banque.",
    "Les frais de tenue de compte ont augmenté de {frais}€ cette année sans aucune notification préalable. Je vais changer de banque.",
    "Le service client téléphonique est injoignable. J'ai attendu {attente} minutes pour finalement être raccroché au nez.",
    "Conseiller d'agence très désagréable qui refuse de m'aider pour mon dossier de prêt immobilier. Zéro professionnalisme.",
    "Ma carte bancaire a été bloquée à l'étranger sans raison alors que j'avais bien prévenu de mon voyage. Je me suis retrouvé sans argent.",
    "J'ai été victime d'une fraude sur ma carte de {frais}€ et la banque refuse de me rembourser en m'accusant de négligence.",
    "L'agence physique près de chez moi est toujours fermée l'après-midi. Impossible de déposer mes chèques.",
    "Crédit refusé sans aucune explication après 3 semaines d'attente et de promesses. Perte de temps totale.",
    "Les taux d'intérêt sur le livret d'épargne sont ridicules. Autant garder son argent sous le matelas.",
    "Nouvelle interface de l'application très confuse et lente, la mise à jour est complètement ratée.",
    # Positifs
    "Très satisfait de mon conseiller qui a su m'accompagner avec beaucoup de réactivité pour mon crédit auto.",
    "L'application est super intuitive et rapide, les virements se font en un clic. Je recommande vivement.",
    "Ouverture de compte en ligne simple et rapide en moins de 10 minutes. La carte est arrivée en 3 jours.",
    "Excellent accueil en agence ce matin pour le renouvellement de ma carte de paiement. Personnel souriant et efficace.",
    "J'ai eu un problème de paiement double et le service client a réglé le problème et m'a remboursé les frais en 24h. Bravo !",
    "Banque éthique avec des tarifs très clairs et sans frais cachés. C'est rare de nos jours.",
    "Le crédit immobilier a été accepté rapidement avec un taux très compétitif. Un grand merci à toute l'équipe.",
    "Le système d'alertes en cas de transaction suspecte fonctionne à merveille, cela me rassure beaucoup au quotidien.",
    "Le conseiller patrimonial m'a donné d'excellents conseils pour mes placements financiers. Très pro.",
    "Je suis client depuis 15 ans et le service a toujours été d'une régularité exemplaire. Confiance totale.",
    # Neutres
    "Banque correcte, tarifs dans la moyenne des banques traditionnelles. Rien d'exceptionnel mais pas de gros soucis.",
    "Les agences physiques sont propres mais les horaires d'ouverture sont un peu limités pour les gens qui travaillent.",
    "J'utilise cette banque uniquement pour mon compte secondaire, cela convient pour un usage basique.",
    "Le site web fonctionne correctement mais il mériterait une petite mise à jour esthétique.",
    "J'ai reçu ma nouvelle carte bancaire hier comme prévu, mais le code PIN a mis 3 jours de plus à arriver.",
]

# 2. Espagnol (es) - 80 verbatims cibles
ES_TEMPLATES = [
    # Négatifs
    "La aplicación móvil va lentísima y se cierra sola cuando intento ver mis movimientos de tarjeta.",
    "Me han cobrado una comisión de {frais}€ por mantenimiento de cuenta sin previo aviso. Es un robo.",
    "El servicio de atención al cliente por teléfono no responde nunca. Llevo más de {attente} minutos esperando.",
    "Mi tarjeta de crédito fue bloqueada en el extranjero sin ninguna razón. Me dejaron sin dinero en plenas vacaciones.",
    "He sido víctima de un fraude de {frais}€ y el banco no se hace cargo del reembolso. Pésima seguridad.",
    "La oficina de mi barrio ha cerrado y ahora tengo que desplazarme {attente} km para cualquier gestión.",
    "Solicité un préstamo personal y tardaron un mes en decirme que no, pidiendo mil papeles inútiles.",
    "Los intereses de los depósitos a plazo fijo son casi inexistentes. No vale la pena ahorrar aquí.",
    # Positifs
    "Muy contento con la atención recibida en la oficina para contratar mi hipoteca. Todo muy claro.",
    "La aplicación de banca móvil es excelente, muy rápida y fácil de usar para Bizum y transferencias.",
    "Apertura de cuenta online súper rápida y sin comisiones de mantenimiento. Tarjeta gratis en casa.",
    "Resolvieron un cargo duplicado en mi tarjeta en menos de 24 horas. Servicio al cliente de diez.",
    "Llevo años con este banco y nunca he tenido ningún problema. Tarifas transparentes y sin sorpresas.",
    "El sistema de notificaciones de gastos es inmediato y ayuda mucho a controlar mi presupuesto.",
    "Excelente asesoramiento sobre planes de pensiones e inversión. Profesionales de confianza.",
    "La atención en ventanilla es siempre rápida y con personal muy amable.",
    # Neutres
    "Un banco normal, las comisiones están en la media del mercado. Ni el mejor ni el peor.",
    "La tarjeta funciona bien pero la web se cae de vez en cuando los fines de semana.",
    "Tengo la cuenta nómina con ellos porque me obligaba la empresa, pero el servicio es básico.",
    "Me enviaron la nueva tarjeta a tiempo, pero el proceso de activación telefónica fue un poco pesado.",
]

# 3. Portugais (pt) - 80 verbatims cibles
PT_TEMPLATES = [
    # Négatifs
    "O aplicativo do banco trava constantemente ao tentar realizar um Pix. Muito frustrante.",
    "Fui cobrado em uma tarifa de manutenção de {frais}R$ sem nenhum aviso prévio. Vou cancelar a conta.",
    "O atendimento telefônico é horrível. Fiquei esperando {attente} minutos na linha e ninguém atendeu.",
    "Meu cartão de débito foi bloqueado durante uma viagem internacional sem motivo. Fiquei sem dinheiro.",
    "Clonaram meu cartão e o banco se recusa a estornar o valor de {frais}R$. Segurança péssima.",
    "A agência física da minha cidade fechou e agora preciso resolver tudo por um chat que não funciona.",
    "Demora excessiva para aprovação de crédito. Enviei todos os documentos e não tenho retorno há semanas.",
    "As taxas de rendimento da poupança são ridículas. Não vale a pena deixar dinheiro parado lá.",
    # Positifs
    "Abertura de conta digital muito rápida, sem burocracia e com cartão de crédito aprovado na hora.",
    "O app é maravilhoso e muito fácil de usar. Faço Pix e pagamento de boletos em segundos.",
    "Fui muito bem atendido pelo gerente ao renegociar minha dívida. Condições excelentes.",
    "Estornaram uma cobrança indevida no meu cartão de crédito super rápido. Ótimo suporte.",
    "Banco excelente, taxas de juros baixas e sem tarifas abusivas. Recomendo muito.",
    "O seguro do cartão funciona de verdade. Tive um problema e me reembolsaram o valor integral.",
    "O atendimento pelo chat do app é rápido e resolve os problemas sem precisar ir na agência.",
    "Uso o banco para minha empresa e o suporte corporativo é excelente.",
    # Neutres
    "Banco aceitável para o dia a dia. As taxas estão na média e o aplicativo cumpre o papel.",
    "O atendimento na agência é demorado, mas o aplicativo funciona bem e evita que eu precise ir lá.",
    "Tenho conta poupança lá por costume da família, mas os rendimentos não são atraentes.",
    "O cartão de crédito chegou dentro do prazo, mas o limite inicial aprovado foi muito baixo.",
]

# 4. Allemand (de) - 80 verbatims cibles
DE_TEMPLATES = [
    # Négatifs
    "Die Banking-App stürzt ständig ab, wenn ich eine Echtzeitüberweisung tätigen möchte. Unbrauchbar.",
    "Die Kontoführungsgebühren wurden ohne Vorankündigung um {frais}€ erhöht. Das ist unverschämt.",
    "Der telefonische Kundenservice ist eine Katastrophe. Warteschleife von über {attente} Minuten.",
    "Meine Kreditkarte wurde im Urlaub im Ausland grundlos gesperrt. Ich stand komplett ohne Geld da.",
    "Ich bin Opfer eines Kartenbetrugs über {frais}€ geworden und die Bank weigert sich, den Betrag zu erstatten.",
    "Die Filiale vor Ort wurde geschlossen. Jetzt muss ich {attente} Kilometer weit fahren für eine Beratung.",
    "Die Kreditentscheidung dauert viel zu lange. Seit Wochen keine Rückmeldung auf meine Anfrage.",
    "Die Zinsen auf dem Tagesgeldkonto sind ein Witz. Sparen lohnt sich bei dieser Bank überhaupt nicht.",
    # Positifs
    "Sehr kompetente Beratung in der Filiale zum Thema Baufinanzierung. Alle Fragen wurden geklärt.",
    "Die App ist absolut übersichtlich, schnell und sicher. Überweisungen klappen immer problemlos.",
    "Kontoeröffnung online in wenigen Minuten per Video-Ident abgeschlossen. Karte war nach 3 Tagen da.",
    "Freundlicher Kundenservice, der eine Doppelbuchung auf meinem Konto innerhalb von 24 Stunden gelöst hat.",
    "Faire Gebührenstruktur, keine versteckten Kosten. Endlich eine transparente Bank.",
    "Sofortige Push-Nachrichten bei jeder Kartentransaktion helfen mir, den Überblick zu behalten.",
    "Tolle Unterstützung beim Einrichten meines Wertpapierdepots. Sehr kompetente Beratung.",
    "Seit 10 Jahren zufriedener Kunde. Der Service ist stets zuverlässig und professionell.",
    # Neutres
    "Durchschnittliche Bank mit normalen Konditionen. Weder besonders gut noch besonders schlecht.",
    "Die physischen Filialen sind sauber, aber die Öffnungszeiten sind für Berufstätige unpraktisch.",
    "Ich nutze das Konto nur als Nebenkonto, für einfache Überweisungen reicht es völlig aus.",
    "Die Karte kam pünktlich an, aber der PIN-Brief ließ noch einige Tage auf sich warten.",
]

# 5. Belge (fr_BE / nl_BE avec belgicismes) - 60 verbatims cibles
BE_TEMPLATES = [
    # Négatifs
    "L'application Bancontact crash systématiquement quand je scanne le code QR. C'est ennuyeux.",
    "J'ai payé {frais}€ de frais de dossier pour mon prêt, c'est nonante fois trop cher pour ce service.",
    "Service client injoignable sur mon GSM, j'ai sonné septante fois sans réponse. Service lamentable.",
    "Le terminal Bancontact n'a pas fonctionné au magasin et ma carte a été bloquée. Bloqué à la caisse !",
    "Impossible de faire un virement instantané sur le compte d'un autre client belge, le système est bloqué.",
    "Mon conseiller de l'agence de Bruxelles s'est montré très discourtois lors de notre rendez-vous hier.",
    "Les frais de retrait d'argent aux distributeurs neutres (Batopin) sont exagérés.",
    "Mon GSM ne reçoit plus le code d'activation de la banque par SMS, je ne peux plus me connecter à l'app.",
    # Positifs
    "Super virement instantané via l'application Bancontact sur le GSM de mon ami, fait en une seconde !",
    "Le conseiller d'agence à Namur a été très aimable et m'a expliqué les prêts immobiliers en détail.",
    "Très bonne expérience avec le guichet automatique Batopin de ma commune, rapide et sécurisé.",
    "Remboursement de frais indus effectué rapidement après réclamation par e-mail. Bon service après-vente.",
    "L'ouverture de mon compte d'épargne en ligne s'est faite en deux temps trois mouvements, super.",
    "J'apprécie la gratuité de la carte de débit Bancontact et des retraits d'argent.",
    "Service en agence impeccable pour les professionnels, interlocuteur dédié et très réactif.",
    "Le nouveau design de l'application mobile belge est très réussi et agréable à utiliser au quotidien.",
    # Neutres
    "Une banque classique en Belgique. Tarifs honnêtes et couverture de distributeurs correcte.",
    "J'utilise l'application sur mon GSM pour les comptes courants, cela fonctionne de manière stable.",
    "Les agences physiques ferment de plus en plus, mais la banque en ligne compense assez bien.",
    "Virement standard reçu sous 24h ouvrables entre deux banques belges. Délai normal.",
]


def expand_templates(templates: list[str], lang: str, target_count: int, prefix: str) -> list[dict]:
    """Multiplie les templates de base de manière réaliste pour atteindre le nombre cible."""
    results = []
    for i in range(target_count):
        template = random.choice(templates)
        
        # Insérer des variations de variables
        frais_val = str(random.choice([5, 10, 15, 29, 50, 120, 250]))
        attente_val = str(random.choice([10, 15, 20, 30, 45, 60]))
        
        text = template.replace("{frais}", frais_val).replace("{attente}", attente_val)
        
        # Assigner une source de façon équilibrée
        source = random.choice(SOURCES)
        
        results.append({
            "id": f"bank_{prefix}_{i}",
            "text": text,
            "source": source,
            "lang": lang
        })
    return results


def main():
    # 1. Charger les avis d'origine
    orig_path = RAW_DIR / "bank_reviews_sample.csv"
    if not orig_path.exists():
        print(f"❌ Fichier d'origine introuvable à : {orig_path}")
        return
        
    df_orig = pd.read_csv(orig_path)
    print(f"📂 Chargé {len(df_orig)} avis en anglais du dataset d'origine.")

    # Prendre 100 avis en anglais de façon aléatoire ou les premiers
    df_en = df_orig.head(100).copy()
    
    # Distribuer les sources de façon variée sur les avis en anglais
    en_records = []
    for idx, row in df_en.iterrows():
        en_records.append({
            "id": f"bank_en_{idx}",
            "text": row["text"],
            "source": random.choice(SOURCES),
            "lang": "en"
        })
        
    # 2. Générer les autres langues
    # 100 Français, 80 Espagnol, 80 Portugais, 80 Allemand, 60 Belge
    fr_records = expand_templates(FR_TEMPLATES, "fr", 100, "fr")
    es_records = expand_templates(ES_TEMPLATES, "es", 80, "es")
    pt_records = expand_templates(PT_TEMPLATES, "pt", 80, "pt")
    de_records = expand_templates(DE_TEMPLATES, "de", 80, "de")
    be_records = expand_templates(BE_TEMPLATES, "fr", 60, "be")  # étiqueté 'fr' car le français est compris par le LLM, mais contient des belgicisme. On peut aussi utiliser "fr_BE" ou "nl_BE"

    # Mélanger les avis belges (certains en français belge, d'autres traduits en néerlandais belge 'nl')
    # Mettons quelques-uns en néerlandais de Belgique (flamand)
    nl_be_texts = [
        "Mijn Bancontact-kaart werkt niet meer in de winkel, de betaling is geweigerd.",
        "Zeer tevreden over de klantenservice in het kantoor te Antwerpen.",
        "De mobiele app crasht telkens als ik een instant overschrijving probeer te doen.",
        "Snelle opening van een gratis zichtrekening online, heel eenvoudig.",
        "De kosten voor geldopname aan Batopin-automaten zijn veel te hoog geworden.",
        "Mijn gsm ontvangt geen activeringscode meer via sms van de bank.",
        "Vriendelijke adviseur die me goed geholpen heeft met mijn autolening.",
        "Ik heb 20 minuten aan de lijn gewacht om uiteindelijk te worden opgehangen.",
        "Bancontact mobiele app op gsm is super handig om snel te betalen."
    ]
    for i, rec in enumerate(be_records):
        if i % 3 == 0:  # 1/3 en néerlandais
            rec["text"] = random.choice(nl_be_texts)
            rec["lang"] = "nl"

    # Assembler toutes les listes
    all_records = en_records + fr_records + es_records + pt_records + de_records + be_records
    
    # Randomiser l'ordre de tout le dataset pour simuler des flux réels mélangés
    random.shuffle(all_records)
    
    # Créer le DataFrame final
    df_multilingual = pd.DataFrame(all_records)
    
    # Forcer la taille à exactement 500 verbatims
    df_multilingual = df_multilingual.head(500)
    
    # Exporter le fichier
    out_path = RAW_DIR / "multilingual_bank_reviews.csv"
    df_multilingual.to_csv(out_path, index=False)
    
    print(f"\n🎉 DATASET MULTILINGUE CRÉÉ : {out_path}")
    print(f"📊 Volume total : {len(df_multilingual)} verbatims")
    print(f"🌍 Distribution des langues :")
    print(df_multilingual["lang"].value_counts())
    print(f"🏷️ Distribution des sources :")
    print(df_multilingual["source"].value_counts())


if __name__ == "__main__":
    main()
