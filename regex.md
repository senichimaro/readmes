# RegExp Object
A regular expression is an object that describes a pattern of characters.

Regular expressions are used to perform pattern-matching and "search-and-replace" functions on text.

## Syntax
~~~
/pattern/modifiers;
~~~

#### Example
~~~
var patt = /w3schools/i

// Example Use
// case-insensitive search for "w3schools" in a string.
var str = "Visit W3Schools";
var patt = /w3schools/i;
var result = str.match(patt);
document.getElementById("demo").innerHTML = result;

> W3Schools
~~~
Example explained:
* /w3schools/i  is a regular expression.
* w3schools  is a pattern (to be used in a search).
* i  is a modifier (modifies the search to be case-insensitive).

For a tutorial about Regular Expressions, read our [JavaScript RegExp Tutorial](https://www.w3schools.com/js/js_regexp.asp).


## Modifiers
Modifiers are used to perform case-insensitive and global searches:

| Modifier |	Description
|-||
g | >	Perform a global match (find all matches rather than stopping after the first match)
i | >	Perform case-insensitive matching
m | >	Perform multiline matching

## Brackets
Brackets are used to find a range of characters:

| Expression |	Description
|-||
[abc]	| > Find any character between the brackets
[^abc]	| > Find any character NOT between the brackets
[0-9]	| > Find any character between the brackets (any digit)
[^0-9]	| > Find any character NOT between the brackets (any non-digit)
(x\|y)	| > Find any of the alternatives specified

## Metacharacters
Metacharacters are characters with a special meaning:

| Metacharacter |	Description
|-||
.	| > Find a single character, except newline or line terminator
\w	| > Find a word character
\W	| > Find a non-word character
\d	| > Find a digit
\D	| > Find a non-digit character
\s	| > Find a whitespace character
\S	| > Find a non-whitespace character
\b	| > Find a match at the beginning/end of a word, beginning like this: \bHI, end like this: HI\b
\B	| > Find a match, but not at the beginning/end of a word
\0	| > Find a NULL character
\n	| > Find a new line character
\f	| > Find a form feed character
\r	| > Find a carriage return character
\t	| > Find a tab character
\v	| > Find a vertical tab character
\xxx	| > Find the character specified by an octal number xxx
\xdd	| > Find the character specified by a hexadecimal number dd
\udddd	| > Find the Unicode character specified by a hexadecimal number dddd

## Quantifiers
| Quantifier |	Description
|-||
n+	| > Matches any string that contains at least one n
n*	| > Matches any string that contains zero or more occurrences of n
n?	| > Matches any string that contains zero or one occurrences of n
n{X}	| > Matches any string that contains a sequence of X n's
n{X,Y}	| > Matches any string that contains a sequence of X to Y n's
n{X,}	| > Matches any string that contains a sequence of at least X n's
n$	| > Matches any string with n at the end of it
^n	| > Matches any string with n at the beginning of it
?=n	| > Matches any string that is followed by a specific string n
?!n	| > Matches any string that is not followed by a specific string n

## RegExp Object Properties
| Property |	Description
|-||
constructor	| > Returns the function that created the RegExp object's prototype
global	| > Checks whether the "g" modifier is set
ignoreCase	| > Checks whether the "i" modifier is set
lastIndex	| > Specifies the index at which to start the next match
multiline	| > Checks whether the "m" modifier is set
source	| > Returns the text of the RegExp pattern

## RegExp Object Methods
| Method  |	Description
|-||
compile()	| > Deprecated in version 1.5. Compiles a regular expression
exec()	| > Tests for a match in a string. Returns the first match
test()	| > Tests for a match in a string. Returns true or false
toString()	| > Returns the string value of the regular expression
