# Knesset election results, 1992 to 2022

knesset_election_results.csv has a row for every list at each of the 13 elections from 1992 to 2022, including lists
that won no seats. It is used for the dealignment chart, which tests dealignment as Shamir, Ventura, Arian and Kedar
(2008) describe it: voters' ties to parties weaken, so votes move a lot between elections, new parties win large
shares and big parties shrink, while the left and right blocs stay much the same.

The results come from the table in each election's English Wikipedia article (the Knesset's own table for 1996 and
1999), read on 23 September 2026. Those tables cite the Central Elections Committee or the Israel Democracy Institute.

The columns are the election date, the list (its English name, with the parts of joint lists), its Wikipedia name,
seats, votes, vote share, lineage (the party family), components (for a joint list, the families in it), bloc and
source URL. Vote share is a percentage of all the lists' votes added together; for 2006 the total is the committee's
figure of 3,137,064. Every election adds up to 120 seats, and the Left's seats match left_seat_share.csv at all 13
elections.

## Party families

Families follow the lineage column of config/factions_en.csv. Families not in that file are named after their list
(tehiya, progressive_list, am_shalem, yachad and zehut). A joint list takes the family of the party that led it:

- Labour: One Israel, Labour-Meimad, the Zionist Union and Labour-Gesher(-Meretz)
- Likud: Likud-Gesher-Tzomet and Likud Yisrael Beiteinu
- Hadash: Hadash-Balad, Hadash-Ta'al and the Joint List
- Ra'am: its lists with Ta'al and with Balad
- national_religious: the National Union (with the NRP in 2006), the Union of Right-Wing Parties and Religious Zionism
- Meretz: the Democratic Union
- blue_white: Blue and White and National Unity
- yamina: Yamina

The components column lists the partners that ran alone at a neighbouring election, for example likud;yisrael_beiteinu
for 2013. Lists without seats keep their family if they have one. The rest, such as lists under about 1% of the vote,
Ale Yarok and the Greens, are grouped as other and counted as one unit.

## Blocs

Blocs follow config/faction_blocs.csv: Left, Secular Centre, Arab-Israeli, Right, Orthodox, and Sectoral for Yisrael
BaAliyah and Gil. A joint list takes the bloc of the party leading it. The NRP (1992 to 2003) and Religious Zionism
(2021 and 2022) are Orthodox. The National Union, the Jewish Home and Yamina (1999 to 2020) and Otzma LeYisrael (2013)
are Right, while Otzma Yehudit running alone (2019 and 2020) is Orthodox. Of the lists without seats that are not in
the config, Tehiya and Zehut are Right, the Progressive List for Peace is Arab-Israeli, Am Shalem and Yachad are
Orthodox, and the rest are Other lists.

## Measures

These are worked out by volatility_table in src/knesset_ches/blog_extra.py.

Volatility is the Pedersen index on votes: half the sum of the changes in vote share from one election to the next,
ignoring their sign.

Between party families, a joint list is compared with the sum of its parts, following Bartolini and Mair. Families
that ran together at either of two elections are treated as one unit for that pair, so a merger or a split does not
count as votes moving.

Between the two camps, Right and Orthodox are set against everyone else, with lists that won no seats and have no bloc
as a third group. These are the camps used in the post and the closest match to the blocs in Shamir et al.

New parties are the seats won by lists whose leading family won no seats at the election before. Kadima (2006), New
Hope (2021), Blue and White in April 2019 (led by Israel Resilience) and Yamina in September 2019 all count as new.

Party-family volatility in 2006 comes to 40.9% (43.5% by leading family), against the 42.7% that Shamir et al.
report. I have not reproduced their calculation exactly.
