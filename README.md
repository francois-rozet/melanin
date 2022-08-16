# Melanin

Melanin is a command-line utility that reformats Python files with the uncompromising [Black](https://github.com/psf/black) formatter. However, it only reformats code that has changed compared to the last [Git](https://git-scm.com/) commit, much like only skin exposed to sunlight gets darker.

> In humans, [melanin](https://wikipedia.org/wiki/Melanin) is the pigment responsible for the skin color and serves as protection against UV radiation. The production of melanin or melanogenesis is initiated by exposure to sunlight, causing the skin to darken. Also, melanin comes from the ancient Greek word for black.

## Installation

The `melanin` package is available on [PyPi](https://pypi.org/project/melanin), which means it is installable via `pip`.

```
pip install melanin
```

Alternatively, if you need the latest features, you can install it using

```
pip install git+https://github.com/francois-rozet/melanin
```

## Getting started

Melanin shamelessly plagiarizes Black's command-line interface. The `tan` command replicates most options provided by `black`, such as `--line-length`, `--diff`,  `--fast` and so on. The only but essential addition is the `--commit` option, which selects the commit to which the working tree is compared. By default, the latter is the current commit (`HEAD`) but any `git checkout` pointer (e.g. `86ccd137`, `HEAD~1`, `dev` or `origin/master`) works.

For example, the following command reformats all Python code in the current directory (`.`) that has changed compared to the penultimate commit of the `master` branch.

```
tan --commit master~1 .
```

And that's it. Enjoy and don't forget to put on sunscreen!

### Configuration

Melanin does not implement its own configuration system, but instead steals [Black's configuration](https://github.com/psf/black#configuration). No need for duplicates!
