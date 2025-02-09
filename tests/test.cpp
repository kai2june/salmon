#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <map>
#include <stdexcept>

void split(std::string line, std::vector<std::string>& ll, char delimiter=' ')
{
    std::istringstream ss(line);
    std::string elem;
    if (delimiter == ' ')
        while( ss >> elem )
            ll.emplace_back(elem);
    else
    {
        while ( std::getline(ss, elem, delimiter) )
            ll.emplace_back(elem);
    }
}

void createNascent(std::string gff3FileName, std::string outputFileName)
{ 
    std::vector<std::vector<std::string>> gene_records;
    std::ifstream ifs(gff3FileName);
    std::string line, last_found, found;
    while(std::getline(ifs, line))
    {
        if (line[0] == '#')
            continue;
        
        if ( line.find("FBgn") != std::string::npos )
        {
            std::vector<std::string> words;
            std::vector<std::string> tags;
            split(line, words);
            split(words.back(), tags, ';');
            for(auto iter=tags.begin(); iter!=tags.end(); ++iter)
                if ( iter->find("geneID=") != std::string::npos )
                    found = iter->substr(iter->find("geneID=") + 7);
            words[2] = "exon";
            words.back() = "Parent=" + found;

            if (last_found == found)
            {
                gene_records.back()[3] = 
                    stoi(gene_records.back()[3]) < stoi(words[3]) ? gene_records.back()[3] : words[3];
                gene_records.back()[4] = 
                    stoi(gene_records.back()[4]) > stoi(words[4]) ? gene_records.back()[4] : words[4];
            }
            else
            {
                gene_records.emplace_back(words);
            }
            last_found = found;
        }
        else
        {
            found.clear();
        }
    }

    /// @brief sort for the sake of unsorted gff3 input file 
    std::sort(gene_records.begin(), gene_records.end(), 
        [](const std::vector<std::string>& lhs, const std::vector<std::string>& rhs)
        {
            return lhs.back() < rhs.back();
        }
    );
    std::vector<std::vector<std::string>> unique_gene_records;
    last_found.clear();
    found.clear();
    for(auto iter=gene_records.begin(); iter!=gene_records.end(); ++iter)
    {
        found = iter->back();
        if (last_found == found)
        {
            unique_gene_records.back()[3] = 
                stoi(unique_gene_records.back()[3]) < stoi((*iter)[3]) ? unique_gene_records.back()[3] : (*iter)[3];
            unique_gene_records.back()[4] = 
                stoi(unique_gene_records.back()[4]) > stoi((*iter)[4]) ? unique_gene_records.back()[4] : (*iter)[4];
        }
        else
        {
            unique_gene_records.emplace_back(*iter);
        }
        last_found = found;
    }
    
    std::sort(unique_gene_records.begin(), unique_gene_records.end(), 
        [](const std::vector<std::string>& lhs, const std::vector<std::string>& rhs)
        {
            if (lhs[0] == rhs[0])
                return stoi(lhs[3]) < stoi(rhs[3]);
            return lhs[0] < rhs[0];
        }
    );

    std::ofstream ofs(outputFileName);
    for(auto iter=unique_gene_records.begin(); iter!=unique_gene_records.end(); ++iter)
    {
        std::vector<std::string> txp_line = (*iter);
        txp_line[2] = "transcript";
        std::string id = txp_line.back().substr(txp_line.back().find("Parent=") + 7);
        txp_line.back() = "ID=" + id + ";geneID=" + id;

        std::string s1, s2;
        for(size_t i=0; i<txp_line.size(); ++i)
        {
            s1 = s1 + txp_line[i];
            s2 = s2 + (*iter)[i];
            if ( i != txp_line.size()-1 )
            {
                s1 = s1 + "\t";
                s2 = s2 + "\t";
            }
            else
            {
                s1 = s1 + "\n";
                s2 = s2 + "\n";
            }
        }
        ofs << s1 << s2;
    }
}

struct GeneInfo
{
    std::string chromosomeName;
    std::string geneName;
    int32_t start;
    int32_t end;
};

void GeneFastaGenerator(std::string genomeFileName, std::string geneMapFileName, std::string outputFastaFileName, std::string txpFileName)
{
    std::ifstream ifs_genome(genomeFileName);
    std::map<std::string, std::string> genome;
    std::string current_chromosome, line;
    while (std::getline(ifs_genome, line))
    {
        if (line[0] == '>')
        {
            std::istringstream ss(line);
            ss >> current_chromosome;
            current_chromosome = current_chromosome.substr(1);
        }
        else
            genome[current_chromosome] += line;
    }
    ifs_genome.close();

    std::vector<GeneInfo> gene_info;
    std::ifstream ifs_geneMap(geneMapFileName);
    while (std::getline(ifs_geneMap, line))
    {
        if(line[0] == '#')
            continue;

        std::vector<std::string> ll;
        split(line, ll);
        if(ll[2] != "transcript")
            continue;
        if( ll[8].find("ID=", 0, 3) == std::string::npos )
            throw std::runtime_error("gff3 record should contain ID!!!");

        gene_info.emplace_back(GeneInfo{});
        gene_info.back().chromosomeName = ll[0];
        gene_info.back().start = stoi(ll[3]);
        gene_info.back().end = stoi(ll[4]);
        std::vector<std::string> ids;
        split(ll[8], ids, ';');
        for(auto elem : ids)
            if (elem.find("ID=", 0, 3) != std::string::npos)
            {
                gene_info.back().geneName = elem.substr(3);
                break;
            }
    }
    ifs_geneMap.close();
    // for (auto elem : gene_info)
    //     std::cerr << "chrname=" << elem.chromosomeName << " start=" << elem.start << " end=" << elem.end << " genename=" << elem.geneName << std::endl;

    std::ofstream ofs_fasta(outputFastaFileName, std::ios::app);
    std::ifstream ifs_txp(txpFileName);
    while(std::getline(ifs_txp, line))
    {
        ofs_fasta << line << "\n";
    }
    for(const auto& elem : gene_info)
    {
        ofs_fasta << ">" << elem.geneName;
        std::string s(genome[elem.chromosomeName].substr(elem.start-1, elem.end-elem.start+1));
        for(int32_t i(0); i<s.size(); ++i)
        {
            if(i%60==0)
                ofs_fasta << "\n";
            ofs_fasta << s[i];
        }
        ofs_fasta << "\n";
    }
    ofs_fasta.close();
    ifs_txp.close();
}

int main()
{
    std::string s1("/home/0309meeting/0413/txp_vs_genetxp/txp/Drosophila_melanogaster.BDGP6.80_txp.gff3"), s2("gene.gff3");
    createNascent(s1, s2);
    std::string genomeFileName("/home/0309meeting/0413/txp_vs_genetxp/txp/Drosophila_melanogaster.BDGP6.dna.chromosome.all.fa");
    std::string outputFastaFileName("genetxp.fa");
    std::string txpFileName("/home/0309meeting/0413/txp_vs_genetxp/Drosophila_melanogaster.BDGP6.80_txp.fa");
    GeneFastaGenerator(genomeFileName, s2, outputFastaFileName, txpFileName);

    return 0;
}