# Voters and seats

The voters and seats chart compares how Jewish Israelis place themselves on left and right with the number of seats
the Left won. Both series are in data/external.

voter_ideology.csv has a row for each survey question and year. The left, centre, right and dont_know columns are
percentages of the Jewish adults asked. Each row also gives the question wording, a link, the fieldwork dates, the
survey mode, the weighting and the sample size. The chart uses the political tendency question from the Israel
National Election Studies (INES). The INES seven-point scale, the Israel Democracy Institute (IDI) series and Pew
are there for comparison. scripts/ines_left_right.py recomputes the INES rows from the public Stata files.

left_seat_share.csv has the Left's seats at each election from 1992 to 2022 (Labour and its joint lists, Meretz and
its lists, and One Nation), with the seats won by Arab lists and by the secular centre. The chart divides the Left's
seats by the seats not won by Arab lists, because the surveys are of Jewish voters. In 2022 Meretz won 3.2% of the
vote, under the 3.25% threshold, and no seats.

## Comparing years

Different questions give different answers. In 2022 the INES tendency question and the IDI scale both put the left at
about 13%, while the INES seven-point scale gives 23%, as many people choose 5, one step left of the middle. In 1992
the two INES questions agreed, at 30.5% and 30.3%. The tendency question was not asked in 2001, September 2019, 2020
or 2021.

The way people were asked also changed. INES interviewed face to face until 1999, by phone from 2001 to 2020 and
online from 2022, and the IDI moved online in 2020. In 2021 INES used both, and the people answering online placed
themselves 7 points further right and were 8 points less likely to say centre. Part of the rise in "right" after 2020
comes from this change. The fall in "left" also shows within the phone years.

INES percentages include people who did not answer, so each row adds up to 100 with dont_know.

In 2006 and 2009 the questions went to only part of the sample, between 340 and 525 Jewish respondents. From 2013 the
tendency question was on one of two versions of the questionnaire, answered by 670 to 720 people. The September 2019
survey re-interviewed a panel and leans left.

INES weights are used from 2019. Earlier Jewish samples are unweighted, as the INES appendices advise.

Most IDI figures were read off the "Political camps" chart in Anabi (2022) and checked against 16 values printed with
it, with a typical error of half a point. Where an IDI appendix table exists (2013, and 2022 to 2025), it has its own
row.

## Sources

- Israel National Election Studies, Tel Aviv University, data and appendices 1988 to 2025.
  https://socsci4.tau.ac.il/mu2/ines/data/our-data/
- Anabi, Or. 2022. "Jewish Israeli Voters Moving Right." Israel Democracy Institute.
  https://en.idi.org.il/articles/45854
- Hermann, Tamar, et al. The Israeli Democracy Index, 2013 to 2025. Israel Democracy Institute.
- Pew Research Center. 2016. Israel's Religiously Divided Society.
- Seats: the IDI's election pages, from the Central Elections Committee results (links in left_seat_share.csv).
