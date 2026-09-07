# Bible-grounded development results

`k=5` · World English Bible (Public Domain) · curated and LLM expansion disabled

## Aggregate

| Configuration | n | Hit@5 | Recall@5 | MRR@5 | nDCG@5 |
|---|---:|---:|---:|---:|---:|
| bm25_title | 66 | 13.6% | 6.8% | 7.8% | 5.7% |
| bm25_bible | 66 | 10.6% | 5.3% | 6.1% | 4.4% |
| dense_title | 66 | 10.6% | 6.8% | 5.7% | 5.3% |
| dense_title_rerank | 66 | 12.1% | 7.6% | 6.3% | 5.7% |
| dense_title_boost | 66 | 12.1% | 7.6% | 6.8% | 6.0% |
| dense_title_full | 66 | 13.6% | 8.3% | 6.4% | 6.1% |
| dense_bible | 66 | 10.6% | 6.1% | 4.3% | 4.2% |
| dense_bible_rerank | 66 | 10.6% | 6.8% | 4.8% | 4.8% |
| dense_bible_boost | 66 | 10.6% | 6.1% | 4.3% | 4.1% |
| dense_bible_full | 66 | 13.6% | 8.3% | 6.0% | 5.8% |

## Abstract

| Configuration | n | Hit@5 | Recall@5 | MRR@5 | nDCG@5 |
|---|---:|---:|---:|---:|---:|
| bm25_title | 53 | 9.4% | 4.7% | 6.3% | 4.4% |
| bm25_bible | 53 | 9.4% | 4.7% | 6.3% | 4.4% |
| dense_title | 53 | 9.4% | 5.7% | 4.3% | 4.1% |
| dense_title_rerank | 53 | 9.4% | 5.7% | 4.4% | 4.2% |
| dense_title_boost | 53 | 9.4% | 5.7% | 3.7% | 3.9% |
| dense_title_full | 53 | 11.3% | 6.6% | 4.2% | 4.4% |
| dense_bible | 53 | 9.4% | 5.7% | 4.3% | 4.1% |
| dense_bible_rerank | 53 | 9.4% | 5.7% | 4.4% | 4.2% |
| dense_bible_boost | 53 | 9.4% | 5.7% | 3.7% | 3.9% |
| dense_bible_full | 53 | 11.3% | 6.6% | 4.2% | 4.4% |

## Explicit Reference

| Configuration | n | Hit@5 | Recall@5 | MRR@5 | nDCG@5 |
|---|---:|---:|---:|---:|---:|
| bm25_title | 1 | 0.0% | 0.0% | 0.0% | 0.0% |
| bm25_bible | 1 | 0.0% | 0.0% | 0.0% | 0.0% |
| dense_title | 1 | 0.0% | 0.0% | 0.0% | 0.0% |
| dense_title_rerank | 1 | 0.0% | 0.0% | 0.0% | 0.0% |
| dense_title_boost | 1 | 0.0% | 0.0% | 0.0% | 0.0% |
| dense_title_full | 1 | 0.0% | 0.0% | 0.0% | 0.0% |
| dense_bible | 1 | 0.0% | 0.0% | 0.0% | 0.0% |
| dense_bible_rerank | 1 | 0.0% | 0.0% | 0.0% | 0.0% |
| dense_bible_boost | 1 | 0.0% | 0.0% | 0.0% | 0.0% |
| dense_bible_full | 1 | 0.0% | 0.0% | 0.0% | 0.0% |

## Narrative

| Configuration | n | Hit@5 | Recall@5 | MRR@5 | nDCG@5 |
|---|---:|---:|---:|---:|---:|
| bm25_title | 2 | 50.0% | 25.0% | 25.0% | 19.3% |
| bm25_bible | 2 | 0.0% | 0.0% | 0.0% | 0.0% |
| dense_title | 2 | 50.0% | 25.0% | 50.0% | 30.7% |
| dense_title_rerank | 2 | 50.0% | 25.0% | 50.0% | 30.7% |
| dense_title_boost | 2 | 50.0% | 25.0% | 50.0% | 30.7% |
| dense_title_full | 2 | 50.0% | 25.0% | 50.0% | 30.7% |
| dense_bible | 2 | 50.0% | 25.0% | 16.7% | 15.3% |
| dense_bible_rerank | 2 | 50.0% | 25.0% | 16.7% | 15.3% |
| dense_bible_boost | 2 | 50.0% | 25.0% | 25.0% | 19.3% |
| dense_bible_full | 2 | 50.0% | 25.0% | 12.5% | 13.2% |

## Quotation

| Configuration | n | Hit@5 | Recall@5 | MRR@5 | nDCG@5 |
|---|---:|---:|---:|---:|---:|
| bm25_title | 10 | 30.0% | 15.0% | 13.3% | 10.8% |
| bm25_bible | 10 | 20.0% | 10.0% | 7.0% | 6.2% |
| dense_title | 10 | 10.0% | 10.0% | 5.0% | 6.9% |
| dense_title_rerank | 10 | 20.0% | 15.0% | 8.3% | 9.6% |
| dense_title_boost | 10 | 20.0% | 15.0% | 15.0% | 13.1% |
| dense_title_full | 10 | 20.0% | 15.0% | 10.0% | 10.8% |
| dense_bible | 10 | 10.0% | 5.0% | 2.5% | 2.6% |
| dense_bible_rerank | 10 | 10.0% | 10.0% | 5.0% | 6.5% |
| dense_bible_boost | 10 | 10.0% | 5.0% | 3.3% | 3.1% |
| dense_bible_full | 10 | 20.0% | 15.0% | 15.0% | 12.6% |

## Grounding audit

| Query | Type | Reference | Confidence |
|---|---|---|---:|
| A Day of Good News | abstract | — | 0.00 |
| Return and Revive | abstract | — | 0.00 |
| The Crises of Rebuilding the Temple (Ezra 4) | explicit_reference | Ezra 4:1-24 | 1.00 |
| Who Returned to Jerusalem? | abstract | — | 0.00 |
| "I am coming soon!", said the Lord | quotation | Revelation 22:4-8 | 0.89 |
| Be Rich Toward God | abstract | — | 0.00 |
| The King with a Tender Heart Revived the Kingdom | abstract | — | 0.00 |
| Shall Two Walk Together, Except They Have Agreed? | quotation | Amos 3:1-5 | 0.63 |
| The Valley of Berachah | abstract | — | 0.00 |
| King Joash | abstract | — | 0.00 |
| The Home of Samuel | abstract | — | 0.00 |
| The Home of Rebekah | abstract | — | 0.00 |
| The Home of Abraham | abstract | — | 0.00 |
| The Home of Noah | abstract | — | 0.00 |
| The Home of Moses | abstract | — | 0.00 |
| The Home of Peter | abstract | — | 0.00 |
| A Joyful Life | abstract | — | 0.00 |
| Knowing that the testing of your faith produces endurance | quotation | James 1:1-5 | 1.00 |
| Identity in Christ | abstract | — | 0.00 |
| Solomon's Prayer at the Temple Dedication | quotation | John 10:19-23 | 0.66 |
| Who is the Greatest | abstract | — | 0.00 |
| Orderly Worshiping | abstract | — | 0.00 |
| Let the Lord Jesus Reign in Our Lives | abstract | — | 0.00 |
| Excel Even More | abstract | — | 0.00 |
| Your Sins Have Been Forgiven | abstract | — | 0.00 |
| The Letter to the Church in Smyrna | abstract | — | 0.00 |
| Rejoicing in Hope | abstract | — | 0.00 |
| Do Not Give the Devil a Foothold | abstract | — | 0.00 |
| What Do You Ask? | abstract | — | 0.00 |
| The Benjamites Who Helped David | abstract | — | 0.00 |
| Do two walk together unless they have agreed to meet? | quotation | Amos 3:1-5 | 0.84 |
| Then he believed in the Lord, and He credited it to him as righteousness | quotation | Genesis 15:4-8 | 0.73 |
| Put on the Helmet of Salvation | abstract | — | 0.00 |
| Turning Our Lives Around: From Pain to Honour | abstract | — | 0.00 |
| Carry the ark of God back into the city! | abstract | — | 0.00 |
| Hezekiah, Who did What Was Right in the Sight of the Lord | quotation | 2 Chronicles 32:22-26 | 0.54 |
| Such Great Faith | abstract | — | 0.00 |
| What is the Reason for the Hope in You? | abstract | — | 0.00 |
| Put on the preparation of the gospel | abstract | — | 0.00 |
| Fifteen Extra Years | abstract | — | 0.00 |
| Joyful Meeting | abstract | — | 0.00 |
| Love is not Jealous | abstract | — | 0.00 |
| Elisha Became Sick | abstract | — | 0.00 |
| Spiritual Significance of Laying of Hands | abstract | — | 0.00 |
| Peace Which Surpasses All Comprehension | narrative | Philippians 4:7-11 | 0.90 |
| My Spirit Rejoices in God | abstract | — | 0.00 |
| The Axe Head Floated | narrative | 2 Kings 6:4-8 | 0.94 |
| Heal the Water at Jericho | abstract | — | 0.00 |
| I Have Learned the Secret | abstract | — | 0.00 |
| Rejoice in the Lord Always | abstract | — | 0.00 |
| The Summit of Pisgah | abstract | — | 0.00 |
| You shall not put the Lord your God to the test | quotation | Matthew 22:34-38 | 0.69 |
| The Ravens Beside the Brook Cherith | quotation | 1 Kings 17:4-8 | 0.74 |
| Elijah prayed for the Rain | abstract | — | 0.00 |
| It is More Blessed to Give than to Receive | abstract | — | 0.00 |
| Receiving Peace in Jesus Christ | quotation | Romans 1:4-8 | 0.85 |
| A Church of Greater Nobility | abstract | — | 0.00 |
| And tongues like fire appeared to them, distributing themselves, and rested on each one of them | abstract | — | 0.00 |
| Succession in the Family | abstract | — | 0.00 |
| From Paul to Timothy | abstract | — | 0.00 |
| Great Wisdom Saved a Small City | abstract | — | 0.00 |
| There is a Man Greater Than Jonah | abstract | — | 0.00 |
| Jesus, Our Lord and King | abstract | — | 0.00 |
| Lord Jesus' Commission - Preach the Gospel | abstract | — | 0.00 |
| Be Joyful in the Hope | abstract | — | 0.00 |
| Do not Leave the Devil a Foothold | abstract | — | 0.00 |
