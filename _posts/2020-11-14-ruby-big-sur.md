---
layout: post
title: Getting "Failed to build gem native extension" error after upgrading to Mac OS Big Sur
date: 2020-12-14 00:00 -0500
---

## Problem

After upgrading to Mac OS Big Sur, I saw this error I haven't seen before:
```bash
➜  bundle exec jekyll serve
Could not find commonmarker-0.17.13 in any of the sources
Run `bundle install` to install missing gems.
```

So I ran `bundle install` but it didn't seem to work either. An excerpt from the terminal output:
```bash
➜  bundle install
Gem::Ext::BuildError: ERROR: Failed to build gem native extension.

An error occurred while installing commonmarker (0.17.13), and Bundler cannot continue.
Make sure that `gem install commonmarker -v '0.17.13' --source 'https://rubygems.org/'` succeeds before bundling.

An error occurred while installing unf_ext (0.0.7.7), and Bundler cannot continue.
Make sure that `gem install unf_ext -v '0.0.7.7' --source 'https://rubygems.org/'` succeeds before bundling.

An error occurred while installing rdiscount (2.2.0.2), and Bundler cannot continue.
Make sure that `gem install rdiscount -v '2.2.0.2' --source 'https://rubygems.org/'` succeeds before bundling.
```


## Solution

After some research and trial and error, I found [this precious answer](https://stackoverflow
.com/a/65017115/9449085) in Stack Overflow that it is due to the Ruby version that is not compatible with Big
Sur and it should be at least 2.7. So I
checked [Ruby releases](https://wwws
.ruby-lang
.org/en/downloads/releases/) and decided to go with one of the most recent releases: 2.7.2.

Anyway, steps I think worked:

1. Check Ruby version
    ```bash
    ➜  ruby -v
    ruby 2.6.3p62 (2019-04-16 revision 67580) [universal.x86_64-darwin20]
    ```
2. Install Ruby Version Manager (rvm)
    ```bash
    ➜  curl -sSL https://raw.githubusercontent.com/rvm/rvm/master/binscripts/rvm-installer | bash -s stable
    ```
3. Install 2.7.2 version using rvm
    ```bash
    ➜  rvm install "ruby-2.7.2"
    ```
4. Check Ruby version again
    ```bash
    ➜  ruby -v
    ruby 2.7.2p137 (2020-10-01 revision 5445e04352) [x86_64-darwin20]
    ```
5. Bundle install
    ```bash
    ➜  bundle install
    ```
6. Run bundle
    ```bash
    ➜  bundle exec jekyll serve
    ```

And it worked!


## Some other things I tried

1. Installing Ruby through Homebrew: didn't solve the issue but I don't know if this actually and eventually helped or 
not.

    ```bash
    ➜  brew install ruby
    ```

2. Installing the latest version of Ruby [(ref)](https://stackoverflow
.com/a/38194139/9449085) using rvm: didn't update the version
 for some reason.

    ```bash
    ➜  rvm install ruby@latest
    ```
