# Matching parties to speech

The Chapel Hill experts rated parties, while the Knesset Corpus records factions. config/party_crosswalk.csv links
the two. For each party the experts rated in 2021 and 2022 it lists the factions (by the corpus's own ids), the dates
and, for a few parties, the members whose speech counts. The speech comes from the calendar year the experts were
rating. config/factions_en.csv gives each corpus faction an English name, a party family and a bloc, and
tests/test_crosswalk.py checks both files.

The 24th Knesset stopped sitting when it dissolved, and its last regular sitting was on 4 July 2022. Speech for 2022
is therefore from January to June, plus the first 21 sitting days of the 25th Knesset, when government and
opposition had changed places. Meretz and Balad won no seats in November 2022, so their 2022 speech is from the 24th
Knesset only.

For the bootstrap each party family is one cluster, which turns the 25 party-years into 12 clusters. The clusters are
numbered in the order the rows appear, which is why Meretz and Balad are listed first. Moving them would shift the
intervals slightly.

## Judgement calls

United Torah Judaism split into Agudat Yisrael (id 18) and Degel HaTorah (83) in August 2022. Both count as UTJ, as
does Yitzhak Pindrus, whose speech the corpus files under Moledet's id (82).

Religious Zionism in 2022 includes Otzma Yehudit (148) and Noam (150) in the 25th Knesset, since the experts rated
the joint list that ran in the 2022 election. Otzma has 13,500 of its 166,000 words and makes little difference.

National Unity (State Camp) in 2022 includes Blue and White (142) and New Hope (146) for the whole year, most of it
from before they merged, so that it rests on more than 21 sitting days.

The Joint List (138) is one party in 2021. In 2022 it is split by speaker. Hadash-Ta'al is its five Hadash and Ta'al
members plus factions 113 and 118, and Balad is Sami Abu Shehadeh alone.

Speech without a faction cannot be matched. This leaves out ministers who gave up their seats under the Norwegian
Law (Elkin, Sa'ar, Horowitz, Zandberg, Hendel and Forer in the 24th Knesset) and members the corpus never linked to a
faction. The main one is Avi Maoz, so Noam has almost no speech.

## Factions and blocs

factions_en.csv has a row for each corpus faction from the 13th to the 25th Knesset. The ids are the corpus's, and a
joint list that kept its senior partner's id shares that row, so 31 also covers Likud-Gesher-Tzomet and Likud Yisrael
Beiteinu, and 35 covers One Israel and the Zionist Union. Renamed parties share a lineage, joint lists with their own
id have their own, and one-member factions are named ind_ and the member's surname. The note column explains the less
obvious calls.

The blocs in the charts come from config/faction_blocs.csv, by faction and Knesset. It follows config/party_blocs.csv,
the party table from my post "Updated Israel map". The bloc in factions_en.csv is used only for factions missing from
faction_blocs.csv.

## Speech filed under the wrong faction

About 0.6% of members' plenary words carry a faction id that does not match the member, because of mistakes in the
corpus. The largest cases are Oren Hazan (20th Knesset, 204,000 words), Ehud Barak (14th, 31,000) and Yehiel Hazan
(16th, 27,000), all under Mapai's old id 12, Uri Ariel (17th, 59,000) under Ra'am-Ta'al (127) instead of National
Union-NRP (126), and Yifat Shasha-Biton (20th, 38,000) under Blue and White (142) instead of Kulanu (137).
faction_blocs.csv puts id 12 on the Left in the 14th Knesset and on the Right in the others, which matches the
parties these members belonged to. Uri Ariel's speech is still counted in the Arab bloc.
