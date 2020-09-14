FROM ubuntu:18.04

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
 git \
 ruby2.5-dev \
 build-essential patch ruby-dev zlib1g-dev liblzma-dev 

RUN gem install json:2.3.1 
RUN gem install nokogiri:1.10.10 
RUN gem install bundler:2.1.4
RUN gem install sass:3.7.4

RUN useradd -m manue
RUN gpasswd -a manue sudo

RUN mkdir /mjai/

WORKDIR /mjai/
RUN git clone https://github.com/gimite/mjai.git
RUN git clone https://github.com/gimite/mjai-manue.git

WORKDIR /mjai/mjai/
ADD mjai.gemspec mjai.gemspec
RUN gem build mjai.gemspec
RUN gem install --local mjai-0.0.7.gem

WORKDIR /mjai/mjai-manue/
RUN gem build mjai-manue.gemspec
RUN gem install --local mjai-manue-0.0.2.gem

WORKDIR /mjai/mjai/
ADD run_manue.sh run_manue.sh

RUN chmod 755 -R /mjai/
RUN chown manue:manue -R /mjai/

USER manue
CMD ["/bin/bash", "./run_manue.sh"]