# GeoPulse — Core Use Case

Source: Competitive Intelligence (CI) use-case brief (slide 3), transcribed verbatim in
structure and content below. This is the reference use case the project is being grounded
against, **broadened per direction (2026-09-22)** — see the note immediately below before
reading the transcribed brief.

## Actual scope (supersedes "Siemens Healthineers" framing below)

This is **not** being built for Siemens specifically — that was the brief's illustrative
example, not the target. The real target is a **consolidated global intelligence system**,
general-purpose, with:

- **Country-level summaries** — per-country rollups: event volume, interaction-type mix
  (cooperation/consultation/conflict), trend over time, top interacting counterpart
  countries/actors.
- **Highlighted events** — a surfaced feed of the most significant events (by media
  attention/mentions, intensity, or anomaly relative to a country's own baseline), not just
  raw event lists.
- **An activity heatmap** — geographic visualization of event volume/intensity, using the
  graph's location data.
- **The general uses already scoped below** — the actor-interaction graph, CAMEO-to-
  interaction-type mapping, and network-pattern analysis over time.

The transcribed brief below still supplies the graph/interaction-type/network-analysis
requirements (WHAT section) and the underlying data approach (HOW/Data sections) — those are
unchanged and still the foundation. Only the *audience and packaging* (a specific company's CI
function) is superseded by the broader system above.

---

## Frame … WHY — Initial Situation and Problem

- In recent years, we have moved away from a world dominated by one or two major powers
  toward a more fragmented and multipolar system, in which multiple centers of power (e.g.
  the United States, China, and the European Union) pursue their own political and economic
  interests.
- This shift creates a more unpredictable environment for businesses, as the decisions of
  these powers increasingly shape markets and the rules of competition.
- From a Competitive Intelligence (CI) perspective, it therefore becomes crucial to monitor
  geopolitical events and identify the underlying drivers and long-term trends in order to
  anticipate how competitors might respond.

## Impact … HOW — Solution, Requirements and Limitations

- To address this problem, we would use the GDELT dataset, which captures and structures
  global political, economic and societal events from international news sources across
  more than 100 languages. By aggregating events based on actors, locations, event types,
  and indicators such as the Goldstein Scale, we can detect patterns relevant for CI.
- **Limitations**: Event and actor information (e.g., generic references such as "company")
  can sometimes be too general to fully capture the specific context of an event *(maybe the
  provided source URLs within the dataset allow further analysis to extract more detailed
  context)*.

## Expected Outcome … WHAT

- Transform GDELT event data into a graph model representing actors and their interactions
  over time.
- Map CAMEO event codes to interpretable interaction types (e.g., cooperation, consultation,
  conflict).
- Analyze the resulting interaction network to identify patterns in geopolitical interactions
  and how they evolve over time.

## Data (Availability)

- The GDELT dataset is publicly available and can be accessed either as downloadable CSV
  files or queried via Google BigQuery.
- The dataset is accompanied by a codebook describing the structure of the data, the meaning
  of each variable, and the coding schemes used to extract actors, locations, themes, and
  sentiment from global news articles.
- Source: GDELT 2.0: Our Global World in Realtime — The GDELT Project.

## Estimated Potential

*(Superseded — see "Actual scope" at the top of this document. Kept here only as the
original brief's own framing, not as current direction.)*

- Beyond Competitive Intelligence, the dataset can support other functions such as Corporate
  Security or Governmental Affairs by analyzing geopolitical developments and identifying
  potential risks for Siemens Healthineers, for example related to supply chain disruptions.
