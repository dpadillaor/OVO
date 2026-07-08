# Inspección visual de splits por-par (miss del contest) — 2026-07-08

Validación a ojo en `contest inspect` (visor 3D) de los miss que pasan `pers>0.5 Y excl>0.75`.
Substrato: `20260708_GT_CLIP_contest-barrier-viz_66656`, `--z-max 1.5`.
Objetivo: confirmar si los miss son transferencias reales (→ eje por-par vale) o smears.
Verde = trozo (split_points) que el challenger roba al defender.

Leyenda veredicto: ✅ bien · ⚠️ dudoso/borde · 🔵 debería-ser-merge (pero split es fix parcial OK)
· ❌ malo/smear · 🔺 pur-baja pero bueno (la pureza GT miente).

---

## office4  (referencia — todos reales)
| par | firm | pers | excl | pur | veredicto ojo |
|---|---:|---:|---:|---:|---|
| 11→15 | 522 | 0.92 | 1.00 | 1.00 | ✅ |
| 82→75 | 311 | 0.83 | 0.93 | 0.95 | ✅ **RESUELTO bueno**: 75 = objeto limpio 20012 (pur 0.99), trozo 95% suyo. 82 es mega-smear (pur 0.35) → contest lo pela: pared→17, objeto→75 |
| 30→45 | 297 | 0.65 | 1.00 | 1.00 | ✅ (el testigo ambiguo suelo/pared → REAL, refina suelo) |
| 17→99 | 137 | 0.58 | 0.82 | 0.94 | ✅ muy nice |
| 36→30 | 43 | 0.72 | 1.00 | 0.93 | ✅ refinamiento de suelo |
| 103→99 | 38 | 0.65 | 1.00 | 0.66 | 🔺 refina objeto muy pequeño MUY preciso (pur miente) |
| 101→99 | 17 | 0.77 | 1.00 | 0.65 | 🔺 igual (pur miente) |
| 75→68 | 16 | 1.00 | 1.00 | 1.00 | ✅ mata puntos flotantes en otra instancia |
| 16→49 | 14 | 0.51 | 1.00 | 0.71 | (techo, no visible — confiamos) |
| 70→30 | 12 | 0.65 | 1.00 | 0.92 | ✅ refina suelo |
| 26→15 | 10 | 0.55 | 1.00 | 1.00 | (techo — confiamos) |
| 4→28 | 9 | 0.59 | 0.78 | 1.00 | (techo) |
| 77→70 | 6 | 0.98 | 1.00 | 1.00 | ✅ refinamiento muy pequeño pero bien |
| 84→77 | 6 | 0.79 | 1.00 | 1.00 | (techo) |
| 13→4 | 5 | 0.53 | 0.80 | 1.00 | (techo) |
| 1→8 | 4 | 1.00 | 1.00 | 1.00 | (demasiado pequeño para ver) |
| 121→30 | 3 | 0.74 | 1.00 | 1.00 | ✅ guay |
| 12→4 | 1 | 0.53 | 1.00 | 1.00 | (techo) |
Resto migajas firm<50: bien pero diminuto.

## office0  ("en general luce todo bien")
| par | firm | pers | excl | pur | veredicto ojo |
|---|---:|---:|---:|---:|---|
| 43→41 | 96 | 1.00 | 1.00 | 0.84 | 🔵 bien, aunque **debería ser merge** |
| 1→15 | 37 | 0.63 | 1.00 | 0.78 | ✅ refinamiento guay |
| 3→1 | 36 | 0.58 | 1.00 | 1.00 | ✅ muy bien |
| 18→19 | 31 | 0.81 | 1.00 | 1.00 | ✅ bien |
| 18→10 | 25 | 0.64 | 1.00 | 0.76 | ⚠️ medio sí medio no |
| 55→47 | 22 | 0.86 | 1.00 | 0.86 | ✅ muy bien |
| 61→56 | 20 | 0.82 | 1.00 | 1.00 | 🔵 bien, **debería ser merge** (merge muy difícil) |
| 47→15 | 15 | 0.51 | 1.00 | 0.80 | ✅ muy guay |
| 15→7 | 9 | 0.59 | 1.00 | 1.00 | ✅ muy bien |
| 103→64 | 7 | 0.55 | 1.00 | 1.00 | ✅ bien |
| 116→3 | 7 | 1.00 | 1.00 | 1.00 | ✅ bien |
| 112→64 | 7 | 0.81 | 1.00 | 1.00 | ✅ bien |
| 24→30 | 6 | 0.59 | 1.00 | 1.00 | ✅ bien |
| 8→114 | 5 | 0.51 | 1.00 | 1.00 | ✅ bien |
| 32→119 | 3 | 0.62 | 1.00 | 1.00 | ✅ bien |
Resto migajas: no comentadas una a una, patrón bueno.

## office3
| par | firm | pers | excl | pur | veredicto ojo |
|---|---:|---:|---:|---:|---|
| 49→44 | 76 | 0.74 | 1.00 | 1.00 | ✅ bien |
| 115→109 | 70 | 0.61 | 0.99 | 0.67 | 🔺 muy bien, mejora mucho un objeto pequeño. pur<1 porque se lleva ALGÚN punto de más (cola sucia real), mayoría buenos → net-positivo pero mejorable |
| 131→103 | 40 | 0.54 | 1.00 | 0.90 | ✅ muy bien, mejora costura alrededor de objeto pequeño |
| 88→43 | 16 | 0.67 | 1.00 | 0.75 | ⚠️ dudosa: costura, puede ser de cualquiera, quizá no debería cambiar |
| 36→0 | 14 | 0.65 | 1.00 | 0.79 | (techo) |
| 107→156 | 13 | 0.52 | 1.00 | 0.85 | (techo) |
| 155→162 | 10 | 0.62 | 1.00 | 1.00 | (techo) |
| 29→10 | 9 | 0.69 | 1.00 | 1.00 | ✅ buena recuperación |
| 43→10 | 9 | 0.56 | 1.00 | 1.00 | ✅ bien |
| 110→43 | 8 | 0.59 | 1.00 | 1.00 | (no visible) |
| 114→124 | 7 | 0.51 | 1.00 | 1.00 | (techo) |
| 107→0 | 6 | 0.54 | 1.00 | 0.83 | (no visible) |
Resto migajas firm<50: demasiado pequeñas para juzgar.
(Nota: `130→103` pur 0.56 y `132→103` pur 0.82 no comentados — cola pequeña.)

## room2   (visor se colgó en 94→93; resto CONFIADO)
| par | firm | pers | excl | pur | veredicto ojo |
|---|---:|---:|---:|---:|---|
| 79→46 | **9496** | 0.64 | 1.00 | 1.00 | ✅ muy bien — **LA BALLENA, masa AP real** |
| 76→72 | 3243 | 0.83 | 1.00 | 0.94 | ✅ bien |
| 76→74 | 1296 | 0.62 | 0.98 | 0.85 | ✅ bien |
| 76→36 | 438 | 0.57 | 0.97 | 0.92 | ✅ bien |
| 37→32 | 315 | 0.58 | 1.00 | 0.98 | ✅ bien |
| 9→6 | 257 | 0.55 | 1.00 | 1.00 | ✅ bien |
| 38→24 | 45 | 0.71 | 0.96 | 1.00 | ✅ bien |
| 36→0 | 19 | 0.96 | 1.00 | 1.00 | ✅ bien |
| 58→32 | 18 | 0.57 | 1.00 | 1.00 | ✅ bien |
| 22→21 | 17 | 1.00 | 1.00 | 0.65 | ✅ costura difícil pero bien (tipo-b: cola sucia, mayoría buenos) |
| 46→7 | 16 | 0.59 | 1.00 | 0.88 | ⚠️ borde: "bien si no pasa también" |
| 16→7 | 12 | 0.66 | 1.00 | 1.00 | ✅ bien |
| 33→24 | 12 | 0.68 | 1.00 | 1.00 | ✅ bien |
| 94→93 | 11 | 0.58 | 1.00 | 0.91 | (visor colgado — CONFIADO) |
Resto migajas firm<50: CONFIADO (no inspeccionado, patrón bueno).

---

## VEREDICTO GLOBAL TUNING (4/4 escenas inspeccionadas)
**Los splits por-par son abrumadoramente REALES.** GO/NO-GO = **GO firme**. La ballena room2
`79→46` (firm 9496) validada a ojo → la masa de AP que focus omitía es real. `pers>0.5 Y excl>0.75`
SIN guard geométrico ya da un set limpio en tuning; los pocos ⚠️ son costuras-borde (`88→43`,
`46→7`, `18→10`), aceptables o net-neutros. Los 🔺 pur-baja son buenos (dos razones: GT miente /
cola sucia recortable). Los 🔵 (`43→41`, `61→56`) son merges-disfrazados, split = fix parcial OK.

---

## Patrones emergentes (a revisar al final)
1. **Los miss son transferencias reales** — GO/NO-GO = GO. pers/excl sin guard geométrico ya
   separa lo bueno de la basura en tuning.
2. **pur baja en split bueno — DOS razones** (🔺): (a) **GT miente** (office4 103/101→99: trozo
   preciso, GT proyectado etiqueta mal); (b) **cola sucia real** (office3 115→109: el trozo se
   lleva unos pocos puntos de más, mayoría buenos). Ambos net-positivos. (a) el guard no debe
   apoyarse en pureza; (b) un guard PODRÍA recortar la cola → mejorable, no descartar.
3. **Split-que-es-merge** (🔵): `43→41`, `61→56`. Containment bajo → bandas merge no lo pillan →
   cae a split; transferir el trozo es fix parcial net-positive, no óptimo. IDEA APARCADA:
   escalar split→merge cuando el resto del defender también sea del challenger.
4. **`82→75` RESUELTO bueno** — 75 = objeto limpio 20012, trozo 95% suyo. 82 es mega-smear (pur 0.35):
   el contest lo PELA en sus objetos reales (pared→17 vía focus, objeto→75 vía por-par). Caso de uso
   canónico del rediseño: un defender suelta varios trozos a varios dueños. (Insight: 82 con pur 0.35
   tiene aún más puntos ajenos → podría beneficiarse de más splits.)
   **NOTA (usuario): 82→75 tiene una PARTICULARIDAD pendiente de comentar. No urge. Revisar al final.**

5. **ETIQUETA MUERTA / dueño fantasma — NO SE DETECTA** (office3, inspección por pares 2026-07-09).
   Un defender puede poseer puntos que **nunca enmascara él mismo**: su lealtad es 0, el 100% de los
   claims son robos de otros. Caso **95** (office3, run perpair-viz_ec060):
   - 114 puntos, todos disputados. Σclaims=693, Σgrabs=693 → **Σleales = 0**.
   - Repartido: 78 roba 388, 13 roba 305 (~55/45). 95 no aparece ni una vez con máscara propia.
   - cont(95,78)=0.272 excl 0.10 ; cont(95,13)=0.167 excl 0.00 → los MISMOS puntos los roban ambos.

   **Por qué no se detecta:** el empate entre dos ladrones destruye la exclusividad (excl≈0) → el split
   no dispara; y ninguno contiene a 95 (cont 0.27/0.17 < low 0.4) → el merge tampoco. 95 sobrevive como
   etiqueta vacía sobre territorio de 78+13 **precisamente porque lo roban dos casi a la vez**. Si lo
   robara uno solo habría sido split/merge y 95 desaparecería. El empate lo deja en el limbo.

   **Es el caso canónico del reeval_frontier** (varios challengers = 1 objeto → sus grabs solapados suman
   a dueño único → disolver 95). Pero reeval está OFF y solo dispara en banda strong (cont≥0.6); aquí los
   cont individuales (0.27/0.17) no llegan → **no aplica ni encendido**. Hueco real: el discriminador no
   tiene ninguna rama para "dueño con lealtad ≈0 disputado por varios sub-strong". La señal existe en el
   store (lealtad = claims−Σgrabs) pero **nadie la mira**. Candidato: gate que detecte lealtad≈0 y fuerce
   reasignación/merge aunque containment individual sea bajo, sumando challengers por raíz. Ver también la
   señal `fiabilidad = claims/sightings` sin explotar (95: claims 693 < sightings 857 → 164 KFs visto sin
   máscara).
