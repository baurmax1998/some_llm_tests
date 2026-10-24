# Snake-Gamer

Steuere Snake zum Futter, ohne Wand oder Körper zu berühren. Der Gamer liefert
bei jedem Zug eine konfigurierbare TCP-Ansicht und zusätzlich den strukturierten
Abschnitt `Exakte Zustandsanalyse`. Die Koordinatenangaben sind maßgeblich.

- `@` = Kopf, `o` = Körper, `*` = Futter.
- Oben links ist `(0,0)`; X wächst nach rechts, Y nach unten.
- Die Kopfzeile nennt aktuelle Richtung und Sekunden pro Tick.
- Eine entgegengesetzte Richtung ist verboten.
- Zum bewussten Geradeausfahren verwende `wait`; wiederhole nicht die aktuelle Richtung.
- Nähert sich der Kopf in seiner Bewegungsrichtung einer Wand, biege rechtzeitig
  in eine erlaubte Richtung ab – auch wenn das Futter woanders liegt.

Der erste Plan wird erzeugt, bevor das Spiel startet. Danach läuft Snake während
der Modellberechnung weiter. Nutze deshalb die angegebene Kollisionsprognose und
die Dauer der vorherigen Antwort. Alte Screenshots im Gespräch sind überholt;
entscheide ausschließlich anhand des neuesten Zuges.

Antworte ausschließlich im anschließend beschriebenen Decoderformat. Zeit wird
als Anzahl vollständiger Spielticks angegeben, niemals in Millisekunden. Plane
nur die nächsten sicheren Aktionen und beende den Plan in einer Richtung mit
mindestens `Benötigter Sicherheitskorridor` freien Ticks. `screenshot` beobachtet
lediglich den Zustand; es startet und stoppt das Spiel nicht. Der Gamer startet
das Spiel automatisch.
