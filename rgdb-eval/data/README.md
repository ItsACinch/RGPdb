# MetaQA data

The harness expects MetaQA files placed here (not committed):

```
data/MetaQA/kb.txt
data/MetaQA/qa_test_1hop.txt
data/MetaQA/qa_test_2hop.txt
data/MetaQA/qa_test_3hop.txt
```

- `kb.txt` lines: `head|relation|tail`
- `qa_test_Nhop.txt` lines: `question with [topic entity]<TAB>answer1|answer2`

Source: the MetaQA dataset (movie KG, ~43k entities, ~135k triples, 9 relations).
Download from the public MetaQA release and copy the vanilla KB + test QA files
into the paths above. The runner (`rgdb_eval.run`) reads these paths by default.
