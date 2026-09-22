# GeoPulse — Core Use Case

Source: Competitive Intelligence (CI) use-case brief (slide 3), transcribed verbatim in
structure and content. This is the reference use case the project is being grounded against.

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

- Beyond Competitive Intelligence, the dataset can support other functions such as Corporate
  Security or Governmental Affairs by analyzing geopolitical developments and identifying
  potential risks for Siemens Healthineers, for example related to supply chain disruptions.
