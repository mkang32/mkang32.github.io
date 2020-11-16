---
layout: post
title: Adding a horizontal scroll to overflowing markdown table in Jekyll
date: 2020-11-15 18:58 -0500
---

## Overflowing table
If you have a wide table, it might overflow your normal post width and look really ugly. It is the case 
for a lot of data 
science projects as there are many features in columns to analyze! For 
example, a table like 
this:


|      | col_name1 | col_name2 | col_name3 | col_name4 | col_name5 | col_name6 | col_name7 | col_name8 | col_name9 | col_name10 |
|------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| row1 |           |           |           |           |           |           |           |           |           |           |
| row2 |           |           |           |           |           |           |           |           |           |           |


Fortunately, I found a way to make such table fit nicely to my Jekyll page layout like this. ([Is there a way to 
overflow a markdown table using HTML?](https://stackoverflow
.com/questions/41076390/is-there-a-way-to-overflow-a-markdown-table-using-html))

<div class="table-wrapper" markdown="block">

|      | col_name1 | col_name2 | col_name3 | col_name4 | col_name5 | col_name6 | col_name7 | col_name8 | col_name9 | col_name10 |
|------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| row1 |           |           |           |           |           |           |           |           |           |           |
| row2 |           |           |           |           |           |           |           |           |           |           |

</div>

<br>

## How to add a horizontal scroll
First, add the following wrapper rule to the css file. 
```css
.table-wrapper {
  overflow-x: scroll;
}
```

And then add this to before and after your table. Make sure you have a blank line between your table and the end 
`</div>` to 
see it in effect. 

```html
<div class="table-wrapper" markdown="block">

</div>
```

Applying this to the above example: 

```html
<div class="table-wrapper" markdown="block">

|      | col_name1 | col_name2 | col_name3 | col_name4 | col_name5 | col_name6 | col_name7 | col_name8 | col_name9 | col_name0 |
|------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| row1 |           |           |           |           |           |           |           |           |           |           |
| row2 |           |           |           |           |           |           |           |           |           |           |

</div>
```
